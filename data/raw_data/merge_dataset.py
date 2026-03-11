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



# --- CONFIGURATION ---
# Use absolute paths to prevent Windows "File Not Found" errors
PROJECT_ROOT = os.getcwd()
MASTER_DIR = os.path.join(PROJECT_ROOT, "final_unified_dataset")
OPEN_IMAGES_SRC = os.path.join(PROJECT_ROOT, "bag_detection_dataset")
ROBOFLOW_SRC = os.path.join(PROJECT_ROOT, "robo_ABOD") 

# Unified Schema: 0: person, 1: bag

# Roboflow Mapping:
# 0 -> 1 (Suspicious bag is still a bag)
# 1 -> 0 (null means the bag is still with person, so we label it as person)
# 2 -> 1 (Suspicious-object is still a bag)
RF_MAP = {0: 1, 1: 0, 2: 1}

# OpenImages Mapping:
# ["Backpack", "Handbag","Suitcase", "Luggage and bags"]
# (Assuming your OI procurement script used these indices)
# 0: Backpack -> 1
# 1: Handbag -> 1
# 2: Suitcase -> 1
# 3: Luggage and bags -> 1
OI_MAP = {0: 1, 1: 1, 2: 1, 3: 1}

def create_yaml(target_path):
    """Generates dataset.yaml in the project root folder"""
    yaml_content = f"""
path: {target_path}
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: bag
"""
    yaml_filename = os.path.join(PROJECT_ROOT, "final_unified_dataset", "dataset.yaml")
    with open(yaml_filename, 'w') as f:
        f.write(yaml_content.strip())
    print(f"✅ Created YAML configuration at: {yaml_filename}")

def process_and_merge(src_path, is_roboflow=True):
    splits = ['train', 'val', 'test']
    # Roboflow sometimes uses 'valid' instead of 'val'
    mapping = RF_MAP if is_roboflow else OI_MAP
    
    for split in splits:
        # Handling Roboflow's 'valid' naming convention
        src_split = "valid" if (is_roboflow and split == "val") else split
        
        if is_roboflow:
            src_img_dir = os.path.join(src_path, src_split, "images")
            src_lbl_dir = os.path.join(src_path, src_split, "labels")
        else:
            src_img_dir = os.path.join(src_path, "images", split)
            src_lbl_dir = os.path.join(src_path, "labels", split)

        target_img_dir = os.path.join(MASTER_DIR, "images", split)
        target_lbl_dir = os.path.join(MASTER_DIR, "labels", split)
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_lbl_dir, exist_ok=True)

        if not os.path.exists(src_lbl_dir): 
            print(f"Skipping {src_split}: directory not found.")
            continue

        for label_file in os.listdir(src_lbl_dir):
            if not label_file.endswith(".txt"): continue
            
            # Copy Image
            img_file = label_file.replace(".txt", ".jpg")
            src_img_path = os.path.join(src_img_dir, img_file)
            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, os.path.join(target_img_dir, img_file))

            # Remap and Write Labels
            with open(os.path.join(src_lbl_dir, label_file), 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.split()
                old_id = int(parts[0])
                if old_id in mapping:
                    parts[0] = str(mapping[old_id])
                    new_lines.append(" ".join(parts) + "\n")
            
            with open(os.path.join(target_lbl_dir, label_file), 'w') as f:
                f.writelines(new_lines)

if __name__ == "__main__":
    if os.path.exists(MASTER_DIR):
        shutil.rmtree(MASTER_DIR)
    
    print("Starting Merge...")
    process_and_merge(OPEN_IMAGES_SRC, is_roboflow=False)
    process_and_merge(ROBOFLOW_SRC, is_roboflow=True)
    
    create_yaml(MASTER_DIR)
    print("Done! You are ready to train.")