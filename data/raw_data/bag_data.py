import fiftyone as fo
import fiftyone.zoo as foz
import os
os.environ["AWS_DEFAULT_REGION"] = "ap-southeast-1"
os.environ["FIFTYONE_SHOW_PROGRESS_BARS"] = "true"
"""
Downloads the Open Images V7 dataset for the specified classes and splits, then exports it in YOLO format.

Train : 800 samples
Validation : 100 samples
Test : 100 samples

Serves to be additional data for our project to make the model more robust and generalizable in detecting other types of bags.

"""


classes = ["Backpack", "Handbag","Suitcase", "Luggage and bags"]
export_dir = "./bag_detection_dataset"


split_map = {
    "train": 800,
    "validation": 100,
    "test": 100
}

for native_split, count in split_map.items():
    print(f"Procuring {count} samples from native Open Images '{native_split}'...")
    
    # Download the specific native split
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
        num_workers=1
    )
    
    # Optional: Delete the local FiftyOne DB entry to save SSD space on your Victus
    fo.delete_dataset(dataset.name)

print(f"Procurement Complete. Data is at {export_dir}")