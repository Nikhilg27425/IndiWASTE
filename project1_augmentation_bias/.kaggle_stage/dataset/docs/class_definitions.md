# Class Definitions — IndiWASTE Dataset

This document defines the 10 waste categories used in the IndiWASTE dataset. Each class contains 300 images (battery: 297), sourced from Google, YouTube, Flickr, Pinterest, Shutterstock, and manual collection.

---

## 1. battery
Household and industrial batteries of all types — AA, AAA, 9V, lithium-ion, button cells, car batteries. Typically cylindrical or rectangular with metallic casing.

**Common confusions:** metal, trash, plastic

---

## 2. biological
Organic and biodegradable waste — food scraps, fruit peels, vegetable waste, garden waste, and other compostable materials.

**Common confusions:** trash, paper, glass

---

## 3. cardboard
Corrugated boxes, packaging cartons, cereal boxes, shipping boxes, and flat cardboard sheets. May appear folded, torn, or stacked.

**Common confusions:** paper, trash, plastic

---

## 4. clothes
Discarded garments, fabric scraps, worn-out textiles, shoes-adjacent fabric items, and wearable accessories made of cloth.

**Common confusions:** trash, shoes, paper

---

## 5. glass
Glass bottles, jars, broken glass, window panes, and transparent or colored glass containers.

**Common confusions:** trash, plastic, metal

---

## 6. metal
Tin cans, aluminum foil, scrap metal, steel containers, metal pipes, and other metallic waste items.

**Common confusions:** trash, glass, plastic

---

## 7. paper
Newspapers, magazines, office paper, paper bags, wrapping paper, and other flat paper-based waste.

**Common confusions:** cardboard, trash, biological

---

## 8. plastic
Plastic bottles, bags, containers, wrappers, straws, and other single-use or multi-use plastic items.

**Common confusions:** trash, glass, metal

---

## 9. shoes
Discarded footwear including sneakers, sandals, boots, and slippers made of rubber, leather, or synthetic materials.

**Common confusions:** clothes, trash, plastic

---

## 10. trash
General mixed waste that does not clearly fall into a single recyclable category — includes multi-material items, contaminated waste, and unidentifiable refuse.

**Common confusions:** biological, paper, plastic

---

## Class Distribution

| Class      | Image Count |
|------------|-------------|
| battery    | 297         |
| biological | 300         |
| cardboard  | 300         |
| clothes    | 300         |
| glass      | 300         |
| metal      | 300         |
| paper      | 300         |
| plastic    | 300         |
| shoes      | 300         |
| trash      | 300         |
| **Total**  | **2997**    |

---

## Dataset Splits

| Split      | Count | Ratio |
|------------|-------|-------|
| Train      | 2097  | 70%   |
| Validation | 449   | 15%   |
| Test       | 451   | 15%   |
