import os
import shutil


"""

This script merges the Open Images V7 data and the Roboflow ABOD dataset into a single unified dataset with a consistent class schema. 
It also generates a dataset.yaml file for easy training with YOLOv8.

Ensure that you have run the procurement script to download the Open Images data and have the Roboflow ABOD dataset ready in the specified folder before executing this merge script.

Download the Roboflow ABOD dataset from: https://universe.roboflow.com/abandoned-object/abandoned-object-detection-9fijk and rename the file to "robo_ABOD" and place it in the project root directory.

After that, run this scrip to create the "final_unified_dataset" folder, which will contain the merged images and labels in the YOLO format, along with a dataset.yaml file for training.

final unified_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml


"""
import os
import shutil

# --- CONFIGURATION ---
PROJECT_ROOT = os.getcwd()
MASTER_DIR = os.path.join(PROJECT_ROOT, "final_unified_dataset")
BAG_OPEN_IMAGES_SRC = os.path.join(PROJECT_ROOT, "bag_detection_dataset")
PERSON_OPEN_IMAGES_SRC = os.path.join(PROJECT_ROOT, "person_detection_dataset")
ROBOFLOW_SRC = os.path.join(PROJECT_ROOT, "robo_ABOD") 

# Unified Schema: 0: person, 1: bag
RF_MAP = {0: 1, 1: 0, 2: 1}
BAG_OI_MAP = {0: 1,1:1,2:1,3:1}  # All bag classes map to 'bag' (1)
PERSON_OI_MAP = {0: 0}

def process_and_merge(src_path, is_roboflow=True, bag=False):
    splits = ['train', 'val', 'test']
    mapping = RF_MAP if is_roboflow else (BAG_OI_MAP if bag else PERSON_OI_MAP)
    
    for split in splits:
        # 1. Handle Roboflow vs FiftyOne naming discrepancies
        src_split = "valid" if (is_roboflow and split == "val") else split
        
        # 2. PATH FIX: FiftyOne usually exports as /labels/train and /images/train
        # Roboflow usually exports as /train/labels and /train/images
        if is_roboflow:
            src_img_dir = os.path.join(src_path, src_split, "images")
            src_lbl_dir = os.path.join(src_path, src_split, "labels")
        else:
            # FiftyOne Style
            src_img_dir = os.path.join(src_path, "images", split)
            src_lbl_dir = os.path.join(src_path, "labels", split)

        # Target dirs (Standard YOLOv8 structure)
        target_img_dir = os.path.join(MASTER_DIR, "images", split)
        target_lbl_dir = os.path.join(MASTER_DIR, "labels", split)
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_lbl_dir, exist_ok=True)

        if not os.path.exists(src_lbl_dir): 
            print(f"Skipping {split}: Source directory not found at {src_lbl_dir}")
            continue

        print(f"Merging {split} from {os.path.basename(src_path)}...")

        for label_file in os.listdir(src_lbl_dir):
            if not label_file.endswith(".txt") or label_file == "classes.txt": 
                continue
            
            # Copy Image
            found_img = False
            for ext in [".jpg", ".jpeg", ".png"]:
                img_file = label_file.replace(".txt", ext)
                src_img_path = os.path.join(src_img_dir, img_file)
                if os.path.exists(src_img_path):
                    shutil.copy(src_img_path, os.path.join(target_img_dir, img_file))
                    found_img = True
                    break

            if not found_img:
                continue

            # Remap and Write Labels
            with open(os.path.join(src_lbl_dir, label_file), 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.split()
                if not parts: continue
                old_id = int(parts[0])
                if old_id in mapping:
                    parts[0] = str(mapping[old_id])
                    new_lines.append(" ".join(parts) + "\n")
            
            # Use 'a' (append) because multiple sources (OI Bag, OI Person, Roboflow) 
            # might contribute to the same image filename if names collide
            with open(os.path.join(target_lbl_dir, label_file), 'w') as f:
                f.writelines(new_lines)

def create_yaml(target_path):
    yaml_content = f"""
path: {target_path}
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: bag
"""
    yaml_filename = os.path.join(MASTER_DIR, "dataset.yaml")
    with open(yaml_filename, 'w') as f:
        f.write(yaml_content.strip())
    print(f"✅ Created YAML at: {yaml_filename}")

if __name__ == "__main__":
    if os.path.exists(MASTER_DIR):
        shutil.rmtree(MASTER_DIR)
    
    print("🚀 Starting Merge...")
    # process_and_merge(BAG_OPEN_IMAGES_SRC, is_roboflow=False, bag=True)
    # process_and_merge(PERSON_OPEN_IMAGES_SRC, is_roboflow=False, bag=False)
    process_and_merge(ROBOFLOW_SRC, is_roboflow=True)
    
    create_yaml(MASTER_DIR)
    print("✨ Done! Your unified dataset is ready for training.")

