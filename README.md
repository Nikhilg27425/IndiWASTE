# IndiWASTE

A multi-class waste classification dataset built as part of a Deep Learning and Computer Vision (DLCV) project. IndiWASTE contains ~3000 labeled images across 10 waste categories, designed to train and evaluate image classification models for automated waste segregation.

---

## Problem Statement

Improper waste disposal is a growing environmental challenge. This project frames waste segregation as a **multi-class image classification problem** — given an image of a waste item, the model must predict which of 10 categories it belongs to. The goal is to build a robust classifier that can assist in real-world waste management pipelines.

---

## Dataset Overview

| Property        | Details                              |
|-----------------|--------------------------------------|
| Total Images    | 2997                                 |
| Classes         | 10                                   |
| Images/Class    | ~300 (battery: 297)                  |
| Image Format    | JPEG (.jpg)                          |
| Annotation Type | Multi-annotator (3 labels per image) |
| Split Ratio     | 70% train / 15% val / 15% test       |

---

## Classes

| Class      | Count | Description                              |
|------------|-------|------------------------------------------|
| battery    | 297   | Household and industrial batteries       |
| biological | 300   | Organic/biodegradable waste              |
| cardboard  | 300   | Boxes, cartons, packaging                |
| clothes    | 300   | Discarded garments and textiles          |
| glass      | 300   | Bottles, jars, broken glass              |
| metal      | 300   | Cans, foil, scrap metal                  |
| paper      | 300   | Newspapers, office paper, bags           |
| plastic    | 300   | Bottles, bags, wrappers                  |
| shoes      | 300   | Footwear of all types                    |
| trash      | 300   | General/mixed unidentifiable waste       |

---

## Repository Structure

```
IndiWASTE/
├── images/                  # Raw images organized by class
│   ├── battery/
│   ├── biological/
│   ├── cardboard/
│   ├── clothes/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   ├── shoes/
│   └── trash/
│
├── metadata/                # Per-class CSV with image source and date info
│   └── <class>_annotations.csv
│
├── annotations/             # Multi-annotator label CSVs
│   └── <class>_labels.csv
│
├── splits/                  # Train / val / test split CSVs
│   ├── train.csv            # 2097 images (70%)
│   ├── val.csv              # 449 images (15%)
│   └── test.csv             # 451 images (15%)
│
└── docs/                    # Dataset documentation
    ├── class_definitions.md
    └── multi_annotator_protocol.md
```

---

## Data Splits

| Split      | Images | Ratio |
|------------|--------|-------|
| Train      | 2097   | 70%   |
| Validation | 449    | 15%   |
| Test       | 451    | 15%   |

Split files (`splits/`) contain three columns: `image_id`, `filename`, `label`.

---

## Annotation Protocol

Each image was independently labeled by 3 annotators. Labels are stored in `annotations/<class>_labels.csv` with columns:

- `img_name` — image filename
- `label1`, `label2`, `label3` — individual annotator labels
- `final_label` — ground truth class

Annotator accuracy was intentionally varied (~50–70%) to simulate real-world annotation noise. See [docs/multi_annotator_protocol.md](docs/multi_annotator_protocol.md) for full details.

---

## Metadata

Each class has a metadata CSV in `metadata/` with:

- `img_id` — unique ID (1–2997)
- `img_name` — filename
- `source` — image origin (google, youtube, flickr, manual, pinterest, shutterstock)
- `date` — collection date (Feb 18 – Mar 13, 2026)
- `label` — class label

Images were sourced primarily from Google (~40%), with the remainder distributed across other platforms.

---

## Documentation

- [Class Definitions](docs/class_definitions.md) — detailed description of each waste class
- [Multi-Annotator Protocol](docs/multi_annotator_protocol.md) — annotation methodology and conflict resolution

---

## Project Context

This dataset was created as part of a **Deep Learning and Computer Vision** course project. The classification task involves:

- Exploring CNN-based architectures (ResNet, EfficientNet, MobileNet, etc.)
- Handling class imbalance and annotation noise
- Evaluating model performance on a held-out test set
- Analyzing inter-class confusion (e.g. glass vs trash, paper vs cardboard)
