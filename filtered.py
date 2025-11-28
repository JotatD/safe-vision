import json
import os
import random
import requests
from typing import List, Dict, Optional
from collections import defaultdict
from PIL import Image
from io import BytesIO

# Set seed at module level for complete reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Configuration

def load_metadata() -> tuple:
    """Load image metadata and split data."""
    print("Loading image metadata...")
    with open('meta_files/image_info/meta.json', 'r') as f:
        image_info = json.load(f)
    
    # Create mapping from image_name to class1
    image_to_class1 = {img['image']: img['class1'] for img in image_info}
    
    # Load all split data
    all_samples = []
    for difficulty in ['easy', 'hardq', 'hards']:
        split_file = f'meta_files/split/{difficulty}.json'
        if os.path.exists(split_file):
            print(f"Loading {difficulty} split...")
            with open(split_file, 'r') as f:
                split_data = json.load(f)
                for sample in split_data:
                    sample['difficulty'] = difficulty
                    all_samples.append(sample)
    
    return image_to_class1, all_samples

def has_valid_refusal(chosen_text: str) -> bool:
    """
    Check if the chosen response contains valid refusal phrases.
    
    Args:
        chosen_text: The chosen response text
        
    Returns:
        True if contains "sorry" or "I cannot" (case-insensitive)
    """
    if not chosen_text:
        return False
    
    chosen_lower = chosen_text.lower()
    
    # Check for refusal patterns
    refusal_patterns = [
        "sorry",
        "i cannot",
        "i can't",
        "i'm unable",
        "i am unable",
        "cannot provide",
        "can't provide",
        "cannot assist",
        "can't assist",
        "cannot help",
        "can't help"
    ]
    
    return any(pattern in chosen_lower for pattern in refusal_patterns)


def save_image_from_dataset(image_name: str, output_folder: str, dataset) -> bool:
    """
    Search for an image by name in the SPA-VL dataset and save it to a folder.
    
    Args:
        image_name: Name of the image to search for (e.g., '0.jpg', '10972.jpg')
        output_folder: Folder to save the image to
        dataset: Pre-loaded dataset
    
    Returns:
        bool: True if image was found and saved, False otherwise
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Search for the image in the dataset
    for entry in dataset:
        if entry.get('image_name') == image_name:
            # Found the image
            image = entry['image']
            
            # Ensure it's a PIL Image
            if not isinstance(image, Image.Image):
                return False
            
            # Save the image
            output_path = os.path.join(output_folder, image_name)
            image.save(output_path)
            return True
    
    return False

from collections import defaultdict
from io import BytesIO
from PIL import Image
import os

# ---------------------------
# Helpers
# ---------------------------

def build_name_index(ds):
    """
    Build a mapping: image_name -> [row_indices].
    Uses column extraction to avoid touching/decoding the image column.
    """
    name_to_idxs = defaultdict(list)
    names = ds["image_name"]  # fast; does not decode images
    for i, name in enumerate(names):
        name_to_idxs[name].append(i)
    return name_to_idxs

def _open_pil_from_dataset_row(ds_nodecode, ds_decode, row_idx):
    """Open image from dataset row, handling both local paths and byte-backed entries."""
    row = ds_nodecode[row_idx]["image"]

    # If it's a dict from decode=False
    if isinstance(row, dict):
        path = row.get("path")
        b = row.get("bytes")

        # Case A: path exists on disk
        if path and os.path.exists(path):
            return Image.open(path).convert("RGB")

        # Case B: open from bytes if path missing
        if b is not None:
            return Image.open(BytesIO(b)).convert("RGB")

    # Case C: fallback to already decoded image
    row2 = ds_decode[row_idx]["image"]
    if isinstance(row2, Image.Image):
        return row2

    return None

# ---------------------------
# Single-image save (replaces your save_image_from_dataset)
# ---------------------------

def save_image_from_dataset(image_name: str, output_folder: str, dataset) -> bool:
    """
    Efficient lookup by image_name using a prebuilt index (built on the fly if absent).
    Avoids decoding the entire dataset.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Build name index once (cheap); in your pipeline you can pass it in if you prefer.
    name_index = build_name_index(dataset)

    idxs = name_index.get(image_name)
    if not idxs:
        return False

    i = idxs[0]  # choose the first if duplicates exist

    try:
        # Use a decode=False view to avoid global decode while iterating
        from datasets import Image as HFImage
        ds_nodecode = dataset.cast_column("image", HFImage(decode=False))
    except Exception:
        ds_nodecode = dataset

    pil = _open_pil_from_dataset_row(ds_nodecode, dataset, i)
    if not isinstance(pil, Image.Image):
        return False

    pil.save(os.path.join(output_folder, image_name))
    return True

# ---------------------------
# Batch save (replaces your save_images_from_dataset_batch)
# ---------------------------

def save_images_from_dataset_batch(image_names, output_folder: str, dataset, name_index=None):
    """
    Save multiple images by name efficiently:
      - Builds/uses a name -> [row_idx] index (no image decode)
      - Uses a decode=False view for lazy path/bytes access
      - Decodes & saves only requested images
    """
    os.makedirs(output_folder, exist_ok=True)

    # Build or reuse the name index
    if name_index is None:
        name_index = build_name_index(dataset)

    # Decode-free dataset view for efficient access to path/bytes
    try:
        from datasets import Image as HFImage
        ds_nodecode = dataset.cast_column("image", HFImage(decode=False))
    except Exception:
        ds_nodecode = dataset

    results = {}

    for name in image_names:
        idxs = name_index.get(name)
        if not idxs:
            results[name] = False
            continue

        i = idxs[0]  # pick first occurrence on duplicates (customize if needed)

        try:
            pil = _open_pil_from_dataset_row(ds_nodecode, dataset, i)
            if isinstance(pil, Image.Image):
                pil.save(os.path.join(output_folder, name))
                results[name] = True
            else:
                results[name] = False
        except Exception:
            results[name] = False

    return results


def create_balanced_subset(image_to_class1: Dict, all_samples: List[Dict],
                          samples_per_category: int = 100) -> List[Dict]:
    """
    Create a balanced subset with specified number of samples per category.
    Only includes samples where "chosen" contains refusal phrases.
    
    Args:
        image_to_class1: Mapping from image name to class1 category
        all_samples: All available samples from splits
        samples_per_category: Number of samples to select per category
        
    Returns:
        Combined list of selected samples with valid refusals
    """
    # Group samples by category and filter for valid refusals
    samples_by_category = defaultdict(list)
    filtered_count = 0
    total_count = 0
    
    for sample in all_samples:
        total_count += 1
        image_name = sample['image']
        class1 = image_to_class1.get(image_name)
        
        if class1:
            # Check if chosen response has valid refusal
            if has_valid_refusal(sample.get('chosen', '')):
                samples_by_category[class1].append(sample)
            else:
                filtered_count += 1
    
    print(f"\nFiltered out {filtered_count} samples without valid refusal phrases")
    print(f"Kept {total_count - filtered_count} samples with refusal phrases")
    
    # Sort samples within each category by image name for consistency
    for category in samples_by_category:
        samples_by_category[category].sort(key=lambda x: x['image'])
    
    # Print category statistics
    print("\n" + "="*60)
    print("Category Statistics (After Filtering):")
    print("="*60)
    for category in sorted(samples_by_category.keys()):
        samples = samples_by_category[category]
        print(f"{category}: {len(samples)} samples available")
    
    # Randomly select samples from each category
    subset_samples = []
    
    print("\n" + "="*60)
    print(f"Selecting up to {samples_per_category} samples per category:")
    print(f"Using random seed: {RANDOM_SEED}")
    print("="*60)
    
    # Process categories in sorted order for consistency
    for category in sorted(samples_by_category.keys()):
        samples = samples_by_category[category]
        
        if len(samples) >= samples_per_category:
            # Create a new Random instance with the seed for this category
            # This ensures consistent selection even if categories are processed differently
            rng = random.Random(RANDOM_SEED + hash(category) % 1000)
            selected = rng.sample(samples, samples_per_category)
        else:
            print(f"Warning: {category} has only {len(samples)} samples, selecting all")
            selected = samples
        
        # Add class1 to each sample
        for sample in selected:
            sample['class1'] = category
        
        subset_samples.extend(selected)
        print(f"{category}: Selected {len(selected)} samples")
    
    # Sort final subset by image name for consistent ordering
    subset_samples.sort(key=lambda x: x['image'])
    
    return subset_samples

def save_subset(subset_samples: List[Dict], output_dir: str, 
                image_folder: str, use_dataset: bool = True):
    """
    Save the subset dataset in JSONL format and save images from dataset or download from URLs.
    
    Args:
        subset_samples: List of selected samples
        output_dir: Directory to save the subset
        image_folder: Folder to save downloaded images
        use_dataset: If True, use HuggingFace dataset to get images. If False, download from URLs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save images
    print("\n" + "="*60)
    print("Saving images...")
    print("="*60)
    failed_count = 0
    
    from datasets import load_dataset
    print("Loading HuggingFace dataset...")
    dataset = load_dataset('sqrti/SPA-VL')['train']

    # Build the index once:
    name_index = build_name_index(dataset)

    # Extract image names and batch save:
    image_names = [sample['image'] for sample in subset_samples]
    print(f"Searching for {len(image_names)} images in dataset...")
    results = save_images_from_dataset_batch(image_names, image_folder, dataset, name_index=name_index)
    
    failed_count = len(results)
    
    # Print progress
    for i, (image_name, success) in enumerate(results.items()):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(results)} images processed...")
        if not success:
            print(f"Warning: Could not find/save {image_name} from dataset")

    print(f"\nSaved  images, {failed_count} failed")
    
    # Save complete subset as JSONL (one JSON object per line)
    output_file = os.path.join(output_dir, 'preferences.jsonl')
    with open(output_file, 'w') as f:
        for sample in subset_samples:
            del sample['id']
            del sample['difficulty']
            del sample['class1']
            sample['image'] = os.path.join(output_dir, sample['image'])
            
            # Ensure consistent key ordering for readability
            json_line = json.dumps(sample, sort_keys=True)
            f.write(json_line + '\n')
    print(f"\nSaved complete subset to {output_file} ({len(subset_samples)} lines)")
    
    # Validate saved samples
    refusal_count = sum(1 for s in subset_samples if has_valid_refusal(s.get('chosen', '')))
    print(f"Validation: {refusal_count}/{len(subset_samples)} samples contain refusal phrases")
    
    # Save statistics
    stats = {
        'random_seed': RANDOM_SEED,
        'total_samples': len(subset_samples),
        'samples_with_refusal': refusal_count,
        'images_failed': failed_count,
        'samples_per_category': {}
    }
    
def main():
    print("="*60)
    print("Creating Balanced Subset from SPA-VL Dataset")
    print("Filtering for samples with refusal phrases")
    print(f"Random Seed: {RANDOM_SEED}")
    print("="*60 + "\n")
    
    # Configuration
    
    # Load metadata
    image_to_class1, all_samples = load_metadata()
    samples_per_category = 1000  # Change this to adjust subset size
    base = f"/output/path/internvl_large_{samples_per_category}"
    output_dir = base
    image_folder = f"{base}/images"
    
    
    print(f"\nTotal samples loaded: {len(all_samples)}")
    
    # Create balanced subset
    subset_samples = create_balanced_subset(
        image_to_class1,
        all_samples,
        samples_per_category
    )
    
    # Save results (this will save images from dataset and save as JSONL)
    # Set use_dataset=True to use HuggingFace dataset, False to download from URLs
    save_subset(subset_samples, use_dataset=True, output_dir=output_dir, image_folder=image_folder)
    print(f"\nTotal samples in subset: {len(subset_samples)}")
    print(f"All samples contain valid refusal phrases in 'chosen' field")
    
    meta = {
    "your-custom-dataset-1": {
      "root": image_folder,
      "annotation": os.path.join(output_dir, 'preferences.jsonl'),
      "data_augment": False,
      "repeat_time": 1,
      "length": len(subset_samples)
      }
    }
    
    meta_path = os.path.join(output_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved meta information to {meta_path}")

if __name__ == "__main__":
    main()
