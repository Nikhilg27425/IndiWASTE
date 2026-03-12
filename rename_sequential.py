import os
from pathlib import Path

# Configuration
target_dir = Path('images')
classes = ['biological', 'clothes', 'glass', 'paper', 'shoes', 'trash']

# Process each class
for class_name in classes:
    class_folder = target_dir / class_name
    
    if not class_folder.exists():
        print(f"Skipping {class_name} - folder not found")
        continue
    
    print(f"\nRenaming images in {class_name}...")
    
    # Get all jpg files and sort them
    image_files = sorted(class_folder.glob('*.jpg'))
    total_images = len(image_files)
    
    print(f"  Found {total_images} images")
    
    # Rename to temporary names first to avoid conflicts
    temp_names = []
    for i, img_file in enumerate(image_files, 1):
        temp_name = class_folder / f"temp_{i}.jpg"
        img_file.rename(temp_name)
        temp_names.append(temp_name)
    
    # Rename to final sequential names
    for i, temp_file in enumerate(temp_names, 1):
        final_name = class_folder / f"{class_name}_{i}.jpg"
        temp_file.rename(final_name)
    
    print(f"  Renamed to {class_name}_1.jpg through {class_name}_{total_images}.jpg")

print("\n✓ Done! All images renamed sequentially.")
