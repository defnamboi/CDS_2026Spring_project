import fiftyone as fo
import fiftyone.zoo as foz
import gc
import os
os.environ["AWS_DEFAULT_REGION"] = "ap-southeast-1"
os.environ["FIFTYONE_SHOW_PROGRESS_BARS"] = "true"
"""
Downloads the Open Images V7 dataset for the Person class and splits, then exports it in YOLO format.

Train : 2618 samples
Validation : 297 samples
Test : 326 samples

Serves to fix class imbalance between Person and Bag, ensuring a 80/10/10 split.
"""

classes = ["Person"]
export_dir = "./person_detection_dataset"

split_map = {
    "train": 2618,
    "validation": 297,
    "test": 326
}

for native_split, count in split_map.items():
    print(f"Procuring {count} Person samples from native Open Images '{native_split}'...")

    try:
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=native_split,
            label_types=["detections"],
            classes=classes,
            max_samples=count,
            only_matching=True
        )

        yolo_split = "val" if native_split == "validation" else native_split

        dataset.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            split=yolo_split,
            classes=classes,
            num_workers=4
        )

        print(f"✓ {count} Person samples exported to '{yolo_split}'")

    except Exception as e:
        print(f"✗ Error on Person / {native_split}: {e}")

    finally:
        fo.delete_dataset(dataset.name)
        del dataset
        gc.collect()

print(f"\nProcurement Complete. Person data is at {export_dir}")