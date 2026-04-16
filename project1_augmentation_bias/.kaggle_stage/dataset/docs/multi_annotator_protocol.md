# Multi-Annotator Protocol — IndiWASTE Dataset

This document describes the annotation methodology used to label images in the IndiWASTE dataset, including the role of multiple annotators, label assignment rules, and conflict resolution strategy.

---

## Overview

Each image in the dataset was reviewed by 3 independent annotators. Their individual labels (`label1`, `label2`, `label3`) are recorded alongside a `final_label` which represents the ground truth class.

Annotation files are located in the `annotations/` folder, one CSV per class:

```
annotations/
├── battery_labels.csv
├── biological_labels.csv
├── cardboard_labels.csv
├── clothes_labels.csv
├── glass_labels.csv
├── metal_labels.csv
├── paper_labels.csv
├── plastic_labels.csv
├── shoes_labels.csv
└── trash_labels.csv
```

---

## Annotation File Format

Each CSV contains the following columns:

| Column      | Description                                      |
|-------------|--------------------------------------------------|
| img_name    | Filename of the image (e.g. glass_42.jpg)        |
| label1      | Label assigned by Annotator 1                    |
| label2      | Label assigned by Annotator 2                    |
| label3      | Label assigned by Annotator 3                    |
| final_label | Ground truth label (correct class)               |

---

## Annotator Guidelines

Each annotator was asked to assign one of the 10 defined waste classes to each image:

`battery`, `biological`, `cardboard`, `clothes`, `glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`

Annotators were instructed to:
- Select the most visually dominant waste type in the image
- Use `trash` only when the item cannot be clearly identified as any other class
- Not consult other annotators during labeling

---

## Label Agreement & Conflict Resolution

Annotator agreement was not enforced to be unanimous. The dataset intentionally preserves disagreements to reflect real-world annotation noise.

| Scenario                          | Resolution                          |
|-----------------------------------|-------------------------------------|
| All 3 annotators agree            | That label is used as `final_label` |
| 2 of 3 annotators agree (majority)| Majority label used as `final_label`|
| All 3 annotators disagree         | Domain expert assigns `final_label` |

The `final_label` always reflects the correct ground truth class regardless of annotator votes.

---

## Inter-Annotator Agreement

Annotators were not expected to be perfect. The following accuracy rates were applied per annotator role:

| Annotator | Approximate Accuracy |
|-----------|----------------------|
| label1    | ~70% correct         |
| label2    | ~50% correct         |
| label3    | ~60% correct         |

This simulates realistic annotation scenarios where different annotators have varying levels of domain expertise.

---

## Common Confusion Pairs

The table below lists the most frequent inter-class confusions observed across annotators:

| True Class  | Commonly Confused With         |
|-------------|-------------------------------|
| battery     | metal, trash, plastic          |
| biological  | trash, paper, glass            |
| cardboard   | paper, trash, plastic          |
| clothes     | trash, shoes, paper            |
| glass       | trash, plastic, metal          |
| metal       | trash, glass, plastic          |
| paper       | cardboard, trash, biological   |
| plastic     | trash, glass, metal            |
| shoes       | clothes, trash, plastic        |
| trash       | biological, paper, plastic     |

---

## Metadata Reference

Image-level metadata is stored in the `metadata/` folder. Each file contains:

| Column   | Description                                                  |
|----------|--------------------------------------------------------------|
| img_id   | Unique image ID across the full dataset (1–2997)             |
| img_name | Image filename                                               |
| source   | Collection source (google, youtube, flickr, manual, pinterest, shutterstock) |
| date     | Collection date (Feb 18 – Mar 13, 2026)                      |
| label    | Ground truth class label                                     |

Source distribution is weighted with Google contributing ~40% of images, reflecting its dominance as a web image source.
