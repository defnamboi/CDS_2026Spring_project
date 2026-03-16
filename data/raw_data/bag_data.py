import fiftyone as fo
import fiftyone.zoo as foz
import os
import gc

os.environ["FIFTYONE_DATASET_ZOO_DIR"] = "D:/fiftyone_zoo"
os.environ["FIFTYONE_DATABASE_DIR"] = "D:/fiftyone_db"
os.environ["AWS_DEFAULT_REGION"] = "ap-southeast-1"

def download_bags_original_labels():
    bag_classes = ["Backpack", "Handbag", "Suitcase", "Luggage and bags"]
    export_dir = "./bag_detection_dataset"
    
    split_map = {
        "test": 100,
        "validation": 200,
        "train": 700
    }

    first_export = True  # ← track whether we've written anything yet

    for native_split, target_obj_count in split_map.items():
        print(f"\n--- Processing {native_split} ({target_obj_count} objects) ---")
        
        try:
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split=native_split,
                label_types=["detections"],
                classes=bag_classes,
                max_samples=target_obj_count,
                only_matching=True,
                shuffle=True,
                seed=42
            )

            view = dataset.filter_labels(
                "ground_truth", fo.ViewField("label").is_in(bag_classes)
            )

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
            yolo_split = "val" if native_split == "validation" else native_split

            final_view.export(
                export_dir=export_dir,
                dataset_type=fo.types.YOLOv5Dataset,
                label_field="ground_truth",
                split=yolo_split,
                classes=bag_classes,
                overwrite=first_export,  # ← True only on first run, False after
            )

            first_export = False  # ← all subsequent splits just append
            print(f"✅ Exported {current_obj_count} objects to {yolo_split} split.")

        except Exception as e:
            print(f"❌ Error: {e}")

        finally:
            if 'dataset' in locals():
                fo.delete_dataset(dataset.name)
            gc.collect()

if __name__ == "__main__":
    download_bags_original_labels()