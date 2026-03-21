import os
import zipfile
import json
from datetime import datetime


"""

This script zips the unified dataset for easy sharing and deployment.

"""


def zip_dataset(source_dir, output_filename):
    # 1. Create a simple manifest for your EDA records
    manifest = {
        "project": "Suspicious Bag Detection",
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "classes": ["Backpack", "Handbag", "Suitcase", "Luggage and bags"],
        "format": "YOLOv5/v8",
        "split_ratio": "80/10/10"
    }
    
    manifest_path = os.path.join(source_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    # 2. Perform the Zipping
    print(f"Compressing {source_dir} into {output_filename}...")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                # Create a relative path to keep the internal folder structure clean
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(source_dir))
                zipf.write(file_path, arcname)
    
    print(f"Successfully created {output_filename}")
    # Optional: Clean up manifest from the local folder if you only want it in the zip
    os.remove(manifest_path)

if __name__ == "__main__":
    # Ensure this matches the export_dir from your procurement script
    DATASET_PATH = "./final_unified_dataset" # Change if you want to zip the merged dataset instead
    ZIP_NAME = "suspicious_bag_data_v5.zip"
    
    if os.path.exists(DATASET_PATH):
        zip_dataset(DATASET_PATH, ZIP_NAME)
    else:
        print(f"Error: {DATASET_PATH} not found. Run your procurement script first!")