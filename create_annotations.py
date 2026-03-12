import csv
import random
from pathlib import Path
from datetime import datetime, timedelta

# Configuration
images_dir = Path('images')
annotations_dir = Path('annotations')
classes = ['biological', 'clothes', 'glass', 'paper', 'shoes', 'trash']

# Source options with weights (google gets higher percentage)
sources = ['google', 'youtube', 'flickr', 'manual', 'pinterest', 'shutterstock']
source_weights = [40, 15, 15, 10, 10, 10]  # google has 40% probability

# Date range: Feb 18, 2026 to March 13, 2026
start_date = datetime(2026, 2, 18)
end_date = datetime(2026, 3, 13)
total_days = (end_date - start_date).days + 1

# Create annotations directory
annotations_dir.mkdir(exist_ok=True)

# Process each class
img_id_counter = 1

for class_name in classes:
    class_folder = images_dir / class_name
    
    if not class_folder.exists():
        print(f"Skipping {class_name} - folder not found")
        continue
    
    print(f"\nCreating annotations for {class_name}...")
    
    # Get all jpg files and sort them
    image_files = sorted(class_folder.glob('*.jpg'))
    total_images = len(image_files)
    
    # Create CSV file for this class
    csv_file = annotations_dir / f"{class_name}_annotations.csv"
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['img_id', 'img_name', 'source', 'date', 'label'])
        
        # Generate sequential dates
        date_increment = total_days / total_images
        
        for i, img_file in enumerate(image_files):
            img_id = img_id_counter
            img_name = img_file.name
            source = random.choices(sources, weights=source_weights)[0]
            
            # Calculate date sequentially
            days_offset = int(i * date_increment)
            current_date = start_date + timedelta(days=days_offset)
            date_str = current_date.strftime('%Y-%m-%d')
            
            label = class_name
            
            writer.writerow([img_id, img_name, source, date_str, label])
            img_id_counter += 1
    
    print(f"  Created {csv_file} with {total_images} entries")
    print(f"  Image IDs: {img_id_counter - total_images} to {img_id_counter - 1}")

print(f"\n✓ Done! All annotation files created in '{annotations_dir}' folder.")
