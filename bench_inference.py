import json
import os
from collections import defaultdict
from datetime import datetime

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import argparse

from classification_utils import (
    GEMINI_EVAL_CONTENT_PROMPT,
    GEMINI_EVAL_SYSTEM_PROMPT,
    gemini_api,
    parse_gemini_classification,
)

# ---- InternVL image utilities (adapted from official inference script) ----
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


# ---- Benchmark + InternVL3 chat setup ----
datetime_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
print(datetime_str)

parser = argparse.ArgumentParser(description="Benchmark InternVL model on VLBreakBench.")
parser.add_argument('--model_path', type=str, required=True, help='Path or hub name for the model to use.')
args = parser.parse_args()

category_stats = defaultdict(lambda: {"yes": 0, "no": 0, "unknown": 0})

# Path to your model and benchmark file
model_path = args.model_path
# model_path = "/scratch/jdg8833/Code/InternVL/internvl_chat/work_dirs/internvl_chat_v3_mpo/Internvl3-2B"

json_path = "VLBreakBench/vlbreakbench_base.json"
image_base_path = "VLBreakBench/"  # folder where images are stored

# Load InternVL model + tokenizer
# model = AutoModel.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     use_flash_attn=True,
#     trust_remote_code=True,
# ).cuda()
# model.eval()
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

generation_config = {"max_new_tokens": 512, "do_sample": False}

# Read the benchmark data
with open(json_path, "r") as f:
    bench_data = json.load(f)

# Loop through all prompts
results = []
for entry in bench_data:
    image_path = os.path.join(image_base_path, entry["image"])
    # pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()

    question = f"<image>\n{entry['query']}"
    answer = model.chat(tokenizer, pixel_values, question, generation_config)

    gemini_raw = gemini_api(
        GEMINI_EVAL_SYSTEM_PROMPT, GEMINI_EVAL_CONTENT_PROMPT.format(response=answer)
    )
    classification = parse_gemini_classification(gemini_raw)

    category = entry.get("category", "unknown")
    if classification in ("yes", "no", "unknown"):
        category_stats[category][classification] += 1
    else:
        category_stats[category]["unknown"] += 1

    print(f"Prompt: {entry['query']} (cat: {category})")
    print(f"Answer: {answer}")
    print(
        f"Gemini evaluation: {classification} / Running Tally: {category_stats[category]}\n{'-'*30}"
    )

    results.append(
        {
            "question": entry.get("query", ""),
            "category": category,
            "subcategory": entry.get("subcategory", ""),
            "image": entry.get("image", ""),
            "model_response": answer,
            "gemini_classification": classification,
            "gemini_raw": gemini_raw,
        }
    )

    total_harmful = sum(category_stats[cat]["yes"] for cat in category_stats)
    total_benign = sum(category_stats[cat]["no"] for cat in category_stats)
    total_unknown = sum(category_stats[cat]["unknown"] for cat in category_stats)
    total_total = total_harmful + total_benign + total_unknown

    with open(f"benchmark_model_results_{datetime_str}_{model_path.split('/')[-1]}.json", "w") as f:
        json.dump(
            {
                "results": results,
                "category_stats": dict(category_stats),
                "total_harmful": total_harmful,
                "total_benign": total_benign,
                "total_unknown": total_unknown,
                "total_total": total_total,
            },
            f,
            indent=2,
        )
