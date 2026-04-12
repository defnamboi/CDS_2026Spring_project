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

def download_bags_stable():
    bag_classes = ["Backpack", "Handbag", "Suitcase", "Luggage and bags"]
    export_dir = "./bag_detection_dataset"

    split_map = {
        "test": 200,
        "validation": 200,
        "train": 1600,
    }

    for native_split, target_image_count in split_map.items():
        print(f"\n--- Procuring {target_image_count} single-bag images from '{native_split}' ---")

        try:
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split=native_split,
                label_types=["detections"],
                classes=bag_classes,
                max_samples=target_image_count * 4,  # Wider pool to find enough single-bag images
                only_matching=True,
                shuffle=True,
                seed=42
            )

            # Filter to only bag labels
            view = dataset.filter_labels("ground_truth", fo.ViewField("label").is_in(bag_classes))

            # Keep only images with EXACTLY 1 bag → 1 image = 1 object
            single_bag_view = view.match(
                fo.ViewField("ground_truth.detections").length() == 1
            )

            # Collect up to target_image_count sample IDs
            ids_to_keep = []
            for sample in single_bag_view:
                ids_to_keep.append(sample.id)
                if len(ids_to_keep) >= target_image_count:
                    break

            # Fallback: if not enough single-bag images, relax to <= 3 bags
            if len(ids_to_keep) < target_image_count:
                print(f"  Only found {len(ids_to_keep)} single-bag images, relaxing to <=3...")
                relaxed_view = view.match(
                    fo.ViewField("ground_truth.detections").length() <= 3
                ).exclude(ids_to_keep)
                for sample in relaxed_view:
                    ids_to_keep.append(sample.id)
                    if len(ids_to_keep) >= target_image_count:
                        break

            final_view = view.select(ids_to_keep)
            actual_images = len(ids_to_keep)
            actual_objects = sum(
                len(s.ground_truth.detections) for s in final_view
            )
            print(f"  Images: {actual_images}, Objects: {actual_objects}")

            yolo_split = "val" if native_split == "validation" else native_split
            final_view.export(
                export_dir=export_dir,
                dataset_type=fo.types.YOLOv5Dataset,
                label_field="ground_truth",
                split=yolo_split,
                classes=bag_classes,
                num_workers=1
            )

            print(f"✓ Exported {actual_images} images → '{yolo_split}'")

        except Exception as e:
            print(f"✗ Error on bags / {native_split}: {e}")

        finally:
            if 'dataset' in locals():
                fo.delete_dataset(dataset.name)
                del dataset
            gc.collect()

if __name__ == "__main__":
    download_bags_stable()
    print(f"\nProcurement Complete. Bag data is at ./bag_detection_dataset")