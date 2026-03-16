import fiftyone as fo
import fiftyone.zoo as foz
import gc
import os

# Performance and path configurations
os.environ["AWS_DEFAULT_REGION"] = "ap-southeast-1"
os.environ["FIFTYONE_SHOW_PROGRESS_BARS"] = "true"
os.environ["FIFTYONE_DATABASE_DIR"] = "D:/fiftyone_db" 

fo.config.show_progress_bars = True
fo.config.desktop_app_close_on_exit = True
fo.config.dataset_zoo_dir = "D:/fiftyone_zoo"

def download_persons_stable():
    classes = ["Person"]
    export_dir = "./person_detection_dataset"

    # Stability Trick: Process smaller splits (test/val) before the massive 'train' split
    # These represent target bounding box counts
    split_map = {
        "test": 256,
        "validation": 534,
        "train": 2180,
    }

    for native_split, target_obj_count in split_map.items():
        print(f"\n--- Procuring {target_obj_count} Person OBJECTS from '{native_split}' ---")

        try:
            # Added shuffle=True and seed=42 to prevent memory crashes on the 4.8GB manifest
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split=native_split,
                label_types=["detections"],
                classes=classes,
                max_samples=target_obj_count, 
                only_matching=True,
                shuffle=True, 
                seed=42
            )

            # Filter and prepare view
            view = dataset.filter_labels("ground_truth", fo.ViewField("label").is_in(classes))

            # Select samples by counting individual objects
            ids_to_keep = []
            current_obj_count = 0
            
            for sample in view:
                num_objs = len(sample.ground_truth.detections)
                if current_obj_count + num_objs <= target_obj_count:
                    ids_to_keep.append(sample.id)
                    current_obj_count += num_objs
                
                if current_obj_count >= target_obj_count:
                    break
            
            final_view = view.select(ids_to_keep)
            print(f"Final object count for {native_split}: {current_obj_count}")

            # Export in YOLO format
            yolo_split = "val" if native_split == "validation" else native_split
            final_view.export(
                export_dir=export_dir,
                dataset_type=fo.types.YOLOv5Dataset,
                label_field="ground_truth",
                split=yolo_split,
                classes=classes,
                num_workers=1 # Set to 1 for maximum stability on Windows
            )

            print(f"✓ {current_obj_count} Person objects exported to '{yolo_split}'")

        except Exception as e:
            print(f"✗ Error on Person / {native_split}: {e}")

        finally:
            # Force cleanup after every split to keep the HP Victus RAM clear
            if 'dataset' in locals():
                fo.delete_dataset(dataset.name)
                del dataset
            gc.collect()

if __name__ == "__main__":
    download_persons_stable()
    print(f"\nProcurement Complete. Person data is at ./person_detection_dataset")