import csv
from pathlib import Path

metadata_dir = Path('metadata')
output_file = metadata_dir / 'image_metadata.csv'

classes = ['battery', 'biological', 'cardboard', 'clothes', 'glass',
           'metal', 'paper', 'plastic', 'shoes', 'trash']

annotators = ['A', 'B', 'C', 'D', 'E']

# Collect all rows from individual CSVs
all_rows = []
for class_name in classes:
    csv_file = metadata_dir / f"{class_name}_annotations.csv"
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows.append({
                'image_id': row['img_id'],
                'filename': row['img_name'],
                'label':    row['label'],
                'date':     row['date'],
                'source':   row['source'],
            })

total = len(all_rows)
print(f"Total images: {total}")

# Assign annotators equally — cycle through A,B,C,D,E
for i, row in enumerate(all_rows):
    row['annotator'] = annotators[i % 5]

# Write combined CSV
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['image_id', 'filename', 'label', 'annotator', 'date', 'source'])
    writer.writeheader()
    writer.writerows(all_rows)

# Print annotator distribution
from collections import Counter
dist = Counter(r['annotator'] for r in all_rows)
for k, v in sorted(dist.items()):
    print(f"  Annotator {k}: {v} images")

print(f"\n✓ Created {output_file}")
