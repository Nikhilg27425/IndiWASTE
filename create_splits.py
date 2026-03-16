import csv
import random
from pathlib import Path

images_dir = Path('images')
splits_dir = Path('splits')
splits_dir.mkdir(exist_ok=True)

classes = ['battery', 'biological', 'cardboard', 'clothes', 'glass',
           'metal', 'paper', 'plastic', 'shoes', 'trash']

all_images = []

# Collect all images with their labels
for class_name in classes:
    class_folder = images_dir / class_name
    imgs = sorted(class_folder.glob('*.jpg'))
    for img in imgs:
        all_images.append((img.name, class_name))

print(f"Total images: {len(all_images)}")

# Shuffle
random.seed(42)
random.shuffle(all_images)

total = len(all_images)
train_end = int(total * 0.70)
val_end   = train_end + int(total * 0.15)

train_set = all_images[:train_end]
val_set   = all_images[train_end:val_end]
test_set  = all_images[val_end:]

print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

def write_csv(filepath, data, start_id):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'filename', 'label'])
        for i, (filename, label) in enumerate(data):
            writer.writerow([start_id + i, filename, label])
    return start_id + len(data)

next_id = write_csv(splits_dir / 'train.csv', train_set, 1)
next_id = write_csv(splits_dir / 'val.csv',   val_set,   next_id)
write_csv(splits_dir / 'test.csv',  test_set,  next_id)

print("✓ Done! Split CSVs created in splits/")
