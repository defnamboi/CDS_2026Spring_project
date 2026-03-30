import fiftyone as fo
import fiftyone.zoo as foz
import os, shutil, random
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = r"C:\Users\HP Victus\CDS_2026Spring_project\final_unified_dataset"
os.environ["AWS_DEFAULT_REGION"] = "ap-southeast-1"
os.environ["FIFTYONE_SHOW_PROGRESS_BARS"] = "true"
os.environ["FIFTYONE_DATABASE_DIR"] = "D:/fiftyone_db"

fo.config.show_progress_bars = True
fo.config.desktop_app_close_on_exit = True
fo.config.dataset_zoo_dir = "D:/fiftyone_zoo"
YOLO_CLASSES = {
    "person":   0,
    "handbag":  1,
    "backpack": 1,
    "suitcase": 1,
}
BAG_LABELS = {"handbag", "backpack", "suitcase"}

TARGET = {
    "person_only":    1000,
    "bag_only":       1500,   # increased — will draw from multiple sources
    "person_and_bag": 2000,
}

SPLIT = {"train": 0.80, "val": 0.10, "test": 0.10}
random.seed(42)

# ── Create output folders ─────────────────────────────────────────────────────
for split in SPLIT:
    Path(f"{OUTPUT_DIR}/images/{split}").mkdir(parents=True, exist_ok=True)
    Path(f"{OUTPUT_DIR}/labels/{split}").mkdir(parents=True, exist_ok=True)

# ── Bucket helper ─────────────────────────────────────────────────────────────
def fill_buckets(dataset, buckets):
    """Iterate dataset and fill buckets up to TARGET counts."""
    for sample in dataset:
        dets       = sample.ground_truth.detections if sample.ground_truth else []
        names      = {d.label.lower() for d in dets}
        has_person = "person" in names
        has_bag    = bool(names & BAG_LABELS)

        if has_person and has_bag and len(buckets["person_and_bag"]) < TARGET["person_and_bag"]:
            buckets["person_and_bag"].append(sample)
        elif has_person and not has_bag and len(buckets["person_only"]) < TARGET["person_only"]:
            buckets["person_only"].append(sample)
        elif has_bag and not has_person and len(buckets["bag_only"]) < TARGET["bag_only"]:
            buckets["bag_only"].append(sample)

        if all(len(buckets[k]) >= TARGET[k] for k in TARGET):
            break

    return buckets

# ── Source 1: COCO 2017 train (large pool) ────────────────────────────────────
print("Loading COCO 2017 train...")
coco_train = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    max_samples=20000,      
)

buckets = {k: [] for k in TARGET}
buckets = fill_buckets(coco_train, buckets)

for k, v in buckets.items():
    print(f"After COCO train  {k:<20}: {len(v)}")

# ── Source 2: COCO 2017 validation (extra bag-only images) ───────────────────
if any(len(buckets[k]) < TARGET[k] for k in TARGET):
    print("\nLoading COCO 2017 validation to fill remaining buckets...")
    coco_val = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        max_samples=2000,
    )
    buckets = fill_buckets(coco_val, buckets)

    for k, v in buckets.items():
        print(f"After COCO val    {k:<20}: {len(v)}")

# ── Source 3: OpenImages — bag-only top-up ────────────────────────────────────
if len(buckets["bag_only"]) < TARGET["bag_only"]:
    still_needed = TARGET["bag_only"] - len(buckets["bag_only"])
    print(f"\nStill need {still_needed} bag-only images — loading OpenImages...")

    oi_dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=["Handbag", "Backpack", "Luggage and bags"],
        max_samples=still_needed * 3,  # 3x buffer to account for filtering
    )

    for sample in oi_dataset:
        if len(buckets["bag_only"]) >= TARGET["bag_only"]:
            break

        dets  = sample.ground_truth.detections if sample.ground_truth else []
        names = {d.label.lower() for d in dets}

        # Map OpenImages labels to our BAG_LABELS format
        oi_bag_labels = {"handbag", "backpack", "luggage and bags"}
        has_bag    = bool(names & oi_bag_labels)
        has_person = "person" in names

        if has_bag and not has_person:
            # Remap OpenImages labels to match YOLO_CLASSES
            for det in dets:
                if det.label.lower() in oi_bag_labels:
                    det.label = "handbag"  # normalize to COCO-compatible label
            buckets["bag_only"].append(sample)

    print(f"After OpenImages  bag_only: {len(buckets['bag_only'])}")

# ── Final bucket summary ──────────────────────────────────────────────────────
print("\n── Bucket totals ───────────────────────────────────")
for k, v in buckets.items():
    print(f"  {k:<20}: {len(v)} / {TARGET[k]}")

# ── Split helper ──────────────────────────────────────────────────────────────
def split_samples(samples):
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

# ── Save helper ───────────────────────────────────────────────────────────────
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

        if not yolo_lines:
            return False

    img_name = f"{prefix}_{idx:04d}.jpg"
    lbl_name = f"{prefix}_{idx:04d}.txt"

    shutil.copy(sample.filepath, f"{OUTPUT_DIR}/images/{split}/{img_name}")
    with open(f"{OUTPUT_DIR}/labels/{split}/{lbl_name}", "w") as f:
        f.write("\n".join(yolo_lines))

    return True

# ── Process each bucket ───────────────────────────────────────────────────────
totals = {split: 0 for split in SPLIT}

bucket_meta = [
    ("person_only",    "per", False),
    ("bag_only",       "bag", False),
    ("person_and_bag", "pab", False),
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

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Final split counts ──────────────────────────────")
for split_name in SPLIT:
    count = len(list(Path(f"{OUTPUT_DIR}/images/{split_name}").glob("*.jpg")))
    print(f"  {split_name:<6}: {count} images")

# ── Write dataset.yaml ────────────────────────────────────────────────────────
yaml_path = Path(OUTPUT_DIR) / "dataset.yaml"
yaml_path.write_text(
    f"path: {OUTPUT_DIR}\n"
    f"train: images/train\n"
    f"val:   images/val\n"
    f"test:  images/test\n\n"
    f"nc: 2\n"
    f"names: ['person', 'bag']\n"
)
print(f"\n✅ dataset.yaml written to {yaml_path}")