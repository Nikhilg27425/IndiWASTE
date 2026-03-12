import os
import random
import shutil
from pathlib import Path

# Configuration
source_dir = Path('archive/original')
target_dir = Path('images')
images_per_class = 300

# Specific classes to process
selected_classes = ['biological', 'clothes', 'glass', 'paper', 'shoes', 'trash']

# Get only the selected class folders
class_folders = [source_dir / class_name for class_name in selected_classes if (source_dir / class_name).is_dir()]

print(f"Found {len(class_folders)} classes")

# Create target directory structure
target_dir.mkdir(exist_ok=True)

# Process each class
for class_folder in class_folders:
    class_name = class_folder.name
    print(f"\nProcessing {class_name}...")
    
    # Get all image files in the class folder
    image_files = [f for f in class_folder.glob('*.jpg')]
    total_images = len(image_files)
    
    print(f"  Found {total_images} images")
    
    # Select random images
    if total_images <= images_per_class:
        selected_images = image_files
        print(f"  Selecting all {total_images} images (less than {images_per_class})")
    else:
        selected_images = random.sample(image_files, images_per_class)
        print(f"  Randomly selected {images_per_class} images")
    
    # Create target class folder
    target_class_dir = target_dir / class_name
    target_class_dir.mkdir(exist_ok=True)
    
    # Copy selected images
    for img in selected_images:
        shutil.copy2(img, target_class_dir / img.name)
    
    print(f"  Copied {len(selected_images)} images to {target_class_dir}")

print("\n✓ Done! All images copied successfully.")
