import fiftyone.zoo as foz
import shutil, random
from pathlib import Path

#  Config 
OUTPUT_DIR = r"C:\Users\HP Victus\CDS_2026Spring_project\final_unified_dataset"

# COCO label → YOLO class ID (person=0, bag=1)
YOLO_CLASSES = {
    "person":   0,
    "handbag":  1,
    "backpack": 2,
    "suitcase": 3,
}
BAG_LABELS = {"handbag", "backpack", "suitcase"}

# Target counts per scenario BEFORE splitting
TARGET = {
    "person_only":   2000,   # person, no bag
    "bag_only":      1500,   # bag(s), no person
    "person_and_bag":3000,   # both present
}

SPLIT = {"train": 0.80, "val": 0.10, "test": 0.10}
random.seed(42)

# Create output folders
for split in SPLIT:
    Path(f"{OUTPUT_DIR}/images/{split}").mkdir(parents=True, exist_ok=True)
    Path(f"{OUTPUT_DIR}/labels/{split}").mkdir(parents=True, exist_ok=True)

#  Download COCO pool
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    max_samples=30000,        # large pool to satisfy all buckets
)

#  Bucket samples
buckets = {k: [] for k in TARGET}

for sample in dataset:
    dets       = sample.ground_truth.detections if sample.ground_truth else []
    names      = {d.label.lower() for d in dets}
    has_person = "person" in names
    has_bag    = bool(names & BAG_LABELS)

    if   has_person and     has_bag and len(buckets["person_and_bag"]) < TARGET["person_and_bag"]:
        buckets["person_and_bag"].append(sample)
    elif has_person and not has_bag and len(buckets["person_only"])    < TARGET["person_only"]:
        buckets["person_only"].append(sample)
    elif has_bag    and not has_person and len(buckets["bag_only"])    < TARGET["bag_only"]:
        buckets["bag_only"].append(sample)
    if all(len(buckets[k]) >= TARGET[k] for k in TARGET):
        break

for k, v in buckets.items():
    print(f"{k:<20}: {len(v)} images")

# Split helper 
def split_samples(samples):
    """Returns {'train': [...], 'val': [...], 'test': [...]}"""
    s = samples[:]
    random.shuffle(s)
    n       = len(s)
    n_train = int(n * SPLIT["train"])
    n_val   = int(n * SPLIT["val"])
    return {
        "train": s[:n_train],
        "val":   s[n_train : n_train + n_val],
        "test":  s[n_train + n_val :],
    }

# Save helper
def save_sample(sample, prefix, idx, split, is_background=False):
    dets       = sample.ground_truth.detections if sample.ground_truth else []
    yolo_lines = []

    if not is_background:
        for det in dets:
            label = det.label.lower()
            if label not in YOLO_CLASSES:
                continue
            cls_id     = YOLO_CLASSES[label]
            x, y, w, h = det.bounding_box
            cx, cy     = x + w / 2, y + h / 2
            if w < 0.01 or h < 0.01:
                continue
            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not yolo_lines:          # skip if no valid boxes remain
            return False

    img_name = f"{prefix}_{idx:04d}.jpg"
    lbl_name = f"{prefix}_{idx:04d}.txt"

    shutil.copy(sample.filepath, f"{OUTPUT_DIR}/images/{split}/{img_name}")
    with open(f"{OUTPUT_DIR}/labels/{split}/{lbl_name}", "w") as f:
        f.write("\n".join(yolo_lines))   # empty string = background sample

    return True

#  Process each bucket 
totals = {split: 0 for split in SPLIT}

bucket_meta = [
    ("person_only",    "per",  False),
    ("bag_only",       "bag",  False),
    ("person_and_bag", "pab",  False),
]

for bucket_key, prefix, is_bg in bucket_meta:
    splits = split_samples(buckets[bucket_key])
    saved  = {s: 0 for s in SPLIT}

    for split_name, samples in splits.items():
        for idx, sample in enumerate(samples):
            ok = save_sample(sample, f"{prefix}_{split_name}", idx, split_name, is_background=is_bg)
            if ok:
                saved[split_name]  += 1
                totals[split_name] += 1

    print(f"\n{bucket_key}")
    for s, n in saved.items():
        print(f"  {s:<6}: {n}")

# Summary 
print("\n── Final split counts ──────────────────────────────")
for split_name in SPLIT:
    count = len(list(Path(f"{OUTPUT_DIR}/images/{split_name}").glob("*.jpg")))
    print(f"  {split_name:<6}: {count} images")

# Write dataset.yaml
class_names = ["person", "handbag", "backpack", "suitcase"]

yaml_path = Path(OUTPUT_DIR) / "dataset.yaml"
yaml_path.write_text(
    f"path: {OUTPUT_DIR}\n"
    f"train: images/train\n"
    f"val:   images/val\n"
    f"test:  images/test\n\n"
    f"nc: {len(class_names)}\n"
    f"names: {class_names}\n"
)

print(f"\n✅ dataset.yaml written with {len(class_names)} classes.")