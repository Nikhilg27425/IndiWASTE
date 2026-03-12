import csv
import random
from pathlib import Path

# Configuration
images_dir = Path('images')
annotations_dir = Path('annotations')
classes = ['biological', 'clothes', 'glass', 'paper', 'shoes', 'trash']

# Define realistic confusion patterns - which classes might be confused with each other
confusion_matrix = {
    'biological': ['trash', 'paper', 'glass'],  # organic waste might look like trash
    'clothes': ['trash', 'shoes', 'paper'],  # fabric might be confused
    'glass': ['trash', 'plastic', 'metal'],  # transparent/shiny items
    'paper': ['cardboard', 'trash', 'biological'],  # flat/thin items
    'shoes': ['clothes', 'trash', 'plastic'],  # fabric/rubber items
    'trash': ['biological', 'paper', 'plastic']  # general waste confusion
}

# Add some classes that aren't in our selection but might appear in confusion
all_possible_classes = classes + ['plastic', 'metal', 'cardboard']

# Create annotations directory
annotations_dir.mkdir(exist_ok=True)

# Process each class
for class_name in classes:
    class_folder = images_dir / class_name
    
    if not class_folder.exists():
        print(f"Skipping {class_name} - folder not found")
        continue
    
    print(f"\nCreating label annotations for {class_name}...")
    
    # Get all jpg files and sort them
    image_files = sorted(class_folder.glob('*.jpg'))
    total_images = len(image_files)
    
    # Create CSV file for this class
    csv_file = annotations_dir / f"{class_name}_labels.csv"
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['img_name', 'label1', 'label2', 'label3', 'final_label'])
        
        for img_file in image_files:
            img_name = img_file.name
            final_label = class_name
            
            # Get confusion candidates for this class
            confusion_candidates = confusion_matrix.get(class_name, ['trash', 'biological'])
            
            # Generate 3 labels (some correct, some confused)
            # 70% chance label1 is correct, 30% confused
            label1 = class_name if random.random() < 0.7 else random.choice(confusion_candidates)
            
            # label2 - mix of correct and confused
            label2 = class_name if random.random() < 0.5 else random.choice(confusion_candidates)
            
            # label3 - more confusion
            label3 = class_name if random.random() < 0.6 else random.choice(confusion_candidates)
            
            # Make sure we have some variety - if all three are the same, change one
            labels = [label1, label2, label3]
            if len(set(labels)) == 1 and labels[0] == class_name:
                # All correct, add some confusion
                labels[random.randint(0, 2)] = random.choice(confusion_candidates)
                label1, label2, label3 = labels
            
            writer.writerow([img_name, label1, label2, label3, final_label])
    
    print(f"  Created {csv_file} with {total_images} entries")

print(f"\n✓ Done! All label annotation files created in '{annotations_dir}' folder.")
