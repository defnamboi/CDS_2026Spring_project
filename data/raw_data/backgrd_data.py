import fiftyone as fo
import fiftyone.zoo as foz
import os
import shutil
from pathlib import Path

# ── 1. Download COCO samples ─────────────────────────────────────────────────
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    max_samples=1000,
)

# ── 2. Filter: keep only images with NO person or handbag ────────────────────
# NOTE: load_zoo_dataset with classes= ONLY downloads those classes
# So we load ALL classes and filter manually instead
background_samples = []

for sample in dataset:
    detections = sample.ground_truth.detections if sample.ground_truth else []
    class_names = [d.label.lower() for d in detections]

    has_person  = any("person"  in c for c in class_names)
    has_handbag = any("handbag" in c or "bag" in c for c in class_names)

    if not has_person and not has_handbag:
        background_samples.append(sample.filepath)

    if len(background_samples) == 200:
        break

print(f"Found {len(background_samples)} clean background images")

# ── 3. Copy to local background_images folder ────────────────────────────────
background_dir = "background_images"
os.makedirs(background_dir, exist_ok=True)

for idx, img_path in enumerate(background_samples):
    dst_path = os.path.join(background_dir, f"background_{idx:04d}.jpg")
    shutil.copy(img_path, dst_path)

print(f"Saved {len(background_samples)} images to '{background_dir}/'")

# ── 4. Use absolute Windows path to avoid PermissionError ────────────────────
# Update this to your actual dataset path
dst_images = r"C:\Users\HP Victus\CDS_2026Spring_project\final_unified_dataset\images\train"
dst_labels = r"C:\Users\HP Victus\CDS_2026Spring_project\final_unified_dataset\labels\train"

os.makedirs(dst_images, exist_ok=True)
os.makedirs(dst_labels, exist_ok=True)

# ── 5. Inject into YOLO dataset ──────────────────────────────────────────────
src_images = "background_images"
count = 0

for img_file in os.listdir(src_images):
    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):

        shutil.copy(
            os.path.join(src_images, img_file),
            os.path.join(dst_images, img_file)
        )

        label_name = Path(img_file).stem + ".txt"
        open(os.path.join(dst_labels, label_name), "w").close()
        count += 1

print(f"✅ Added {count} background images to training set")

# ── 6. Sanity check ───────────────────────────────────────────────────────────
img_count = len(os.listdir(dst_images))
lbl_count = len(os.listdir(dst_labels))

print(f"Total images in train : {img_count}")
print(f"Total labels in train : {lbl_count}")
assert img_count == lbl_count, "⚠️ Mismatch — every image needs a label file!"
print("✅ Sanity check passed!")