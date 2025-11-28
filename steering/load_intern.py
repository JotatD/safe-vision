import os

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# Updated to InternVL3-2B
MODEL_PATH = 'OpenGVLab/InternVL3-2B'
LOCAL_ONLY = os.getenv("HF_LOCAL_ONLY", "1") != "0"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
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
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

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
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def _open_image_rgb(image_file):
    img = Image.open(image_file)
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA").convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img

def load_image(image_file, input_size=448, max_num=12):
    image = _open_image_rgb(image_file)
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_model_and_tokenizer():
    print(f"Loading model from {MODEL_PATH}...")
    device_pref = os.getenv("INTERNVL_DEVICE", "auto").lower()  # auto | cpu | cuda
    flash_pref = os.getenv("INTERNVL_FLASH_ATTN", "auto").lower()  # auto | 1 | 0

    if device_pref == "cpu":
        use_cuda = False
    elif device_pref == "cuda":
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = torch.cuda.is_available()

    # Placement strategy: force single GPU to avoid cross-device mismatches
    target_device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    device_map = None  # no sharding

    # Prefer bfloat16 on CUDA for stability on InternVL3-2B
    dtype = torch.bfloat16 if use_cuda else torch.float32

    if flash_pref == "0":
        use_flash_attn = False
    elif flash_pref == "1":
        use_flash_attn = True
    else:
        use_flash_attn = use_cuda  # default: only on CUDA

    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=use_flash_attn,
        trust_remote_code=True,
        device_map=device_map,
        local_files_only=LOCAL_ONLY
    ).eval()

    if target_device is not None:
        model = model.to(target_device, dtype=dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=LOCAL_ONLY
    )
    # Ensure pad_token is set (some models ship without it)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer
