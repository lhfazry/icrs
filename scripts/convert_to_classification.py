import os
import json
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import shutil
from utils.path_util import ensure_root



# Configuration
INPUT_BASE = "data/detection"
OUTPUT_BASE = "data/classification"
TARGET_SIZE = (128, 128)
PADDING_COLOR = (0, 0, 0)  # Black padding color

# Create output directories
os.makedirs(os.path.join(OUTPUT_BASE, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE, "valid"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE, "test"), exist_ok=True)

def get_paths():
    """Return absolute paths for input/output"""
    return {
        "input": {
            "train": "data/detection/train",
            "valid": "data/detection/valid",
            "test": "data/detection/test"
        },
        "output": {
            "train": "data/classification/train",
            "valid": "data/classification/valid",
            "test": "data/classification/test"
        }
    }

def load_coco_categories(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return {cat['id']: cat['name'] for cat in data['categories']}

def process_split(split_name):
    paths = get_paths()
    input_dir = paths["input"][split_name]  # Fixed: use split_name instead of split
    output_dir = paths["output"][split_name]
    
    # Load COCO annotations
    json_path = os.path.join(input_dir, "_annotations.coco.json")
    with open(json_path) as f:
        coco_data = json.load(f)
    
    # Create category mapping (exclude background)
    categories = {
        cat['id']: cat['name'] 
        for cat in coco_data['categories'] 
        if cat['name'].lower() != 'background'  # Exclude background
    }
    
    # Skip if no valid categories found
    if not categories:
        print(f"No valid categories found (excluding background) in {split_name} split")
        return
    
    # Create directories only for non-background categories
    for cat_name in categories.values():
        os.makedirs(os.path.join(output_dir, cat_name), exist_ok=True)
    
    # Create image path mapping
    image_paths = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Process each annotation (filtering out background)
    for ann in tqdm(coco_data['annotations'], desc=f"Processing {split_name}"):
        # Skip if category is background or not in our filtered categories
        if ann['category_id'] not in categories:
            continue
            
        img_id = ann['image_id']
        cat_id = ann['category_id']
        
        # Get image path and open image
        img_name = image_paths[img_id]
        img_path = os.path.join(input_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            print(f"Could not open {img_path}, skipping...")
            continue
        
        # Get bounding box [x, y, width, height]
        x, y, w, h = ann['bbox']
        
        # Convert to absolute coordinates and expand slightly (10%)
        x1 = max(0, int(x - 0.05 * w))
        y1 = max(0, int(y - 0.05 * h))
        x2 = min(img.width, int(x + w * 1.05))
        y2 = min(img.height, int(y + h * 1.05))
        
        # Crop the image
        crop = img.crop((x1, y1, x2, y2))
        
        # Create square image with padding
        max_dim = max(crop.width, crop.height)
        padded = Image.new('RGB', (max_dim, max_dim), PADDING_COLOR)
        padded.paste(crop, (0, 0))
        
        # Resize to target size
        resized = padded.resize(TARGET_SIZE, Image.BILINEAR)
        
        # Save with unique filename
        base_name = os.path.splitext(img_name)[0]
        output_name = f"{base_name}_{ann['id']}.jpg"
        output_path = os.path.join(output_dir, categories[cat_id], output_name)
        resized.save(output_path, quality=95)

def verify_dataset():
    """Verify that all classes exist in all splits"""
    splits = ['train', 'valid', 'test']
    class_lists = []
    
    for split in splits:
        split_dir = os.path.join(OUTPUT_BASE, split)
        classes = sorted(os.listdir(split_dir))
        class_lists.append(set(classes))
    
    # Check all splits have same classes
    if not all(x == class_lists[0] for x in class_lists):
        print("Warning: Not all splits have the same classes!")
    
    print("\nDataset verification:")
    print(f"Classes: {sorted(class_lists[0])}")
    for split in splits:
        split_dir = os.path.join(OUTPUT_BASE, split)
        count = sum(len(files) for _, _, files in os.walk(split_dir))
        print(f"{split}: {count} images")

if __name__ == "__main__":
    ensure_root()
    
    # Process each dataset split
    for split in ['train', 'valid', 'test']:
        process_split(split)
    
    # Verify the output
    verify_dataset()
    print("\nConversion complete!")