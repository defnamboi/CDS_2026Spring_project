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
        "train": 700,
    }

    first_export = True

    for native_split, target_image_count in split_map.items():
        print(f"\n--- Processing {native_split} (target: {target_image_count} images) ---")

        try:
            # Step 1: Download ALL available matching samples — no max_samples cap
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split=native_split,
                label_types=["detections"],
                classes=bag_classes,
                # NO max_samples — pull everything available for these classes
                only_matching=True,
                shuffle=True,
                seed=42
            )

            print(f"  Total downloaded samples: {len(dataset)}")

            # Step 2: Filter labels to only bag classes
            view = dataset.filter_labels(
                "ground_truth", fo.ViewField("label").is_in(bag_classes)
            )

            # Step 3: Try single-bag images first (images == objects)
            single_view = view.match(
                fo.ViewField("ground_truth.detections").length() == 1
            )
            print(f"  Single-bag images available: {len(single_view)}")

            ids_to_keep = [s.id for s in single_view]  # collect ALL, slice later

            if len(ids_to_keep) >= target_image_count:
                # Enough single-bag images — just take what we need
                ids_to_keep = ids_to_keep[:target_image_count]
                print(f"  Using {len(ids_to_keep)} single-bag images.")
            else:
                # Not enough — fill remainder with <=3 bag images
                print(f"  ⚠ Only {len(ids_to_keep)} single-bag images, filling remainder...")
                relaxed_view = view.match(
                    fo.ViewField("ground_truth.detections").length() <= 3
                ).exclude(ids_to_keep)

                needed = target_image_count - len(ids_to_keep)
                extra_ids = [s.id for s in relaxed_view][:needed]
                ids_to_keep.extend(extra_ids)
                print(f"  After relaxed fill: {len(ids_to_keep)} images.")

            # Step 4: Final fallback — if STILL not enough, take whatever is left
            if len(ids_to_keep) < target_image_count:
                print(f"  ⚠ Still only {len(ids_to_keep)} — using all available images.")
                remaining = view.exclude(ids_to_keep)
                extra_ids = [s.id for s in remaining][:target_image_count - len(ids_to_keep)]
                ids_to_keep.extend(extra_ids)

            final_view = view.select(ids_to_keep)
            actual_images = len(ids_to_keep)
            actual_objects = sum(len(s.ground_truth.detections) for s in final_view)
            print(f"  ✓ Final — Images: {actual_images}, Objects: {actual_objects}")

            # Warn if we still came up short (split simply doesn't have enough data)
            if actual_images < target_image_count:
                print(f"  ⚠ WARNING: Split only has {actual_images} bag images total — target unreachable.")

            yolo_split = "val" if native_split == "validation" else native_split
            final_view.export(
                export_dir=export_dir,
                dataset_type=fo.types.YOLOv5Dataset,
                label_field="ground_truth",
                split=yolo_split,
                classes=bag_classes,
                overwrite=first_export,
            )

            first_export = False
            print(f"✅ Exported {actual_images} images to '{yolo_split}' split.")

        except Exception as e:
            print(f"❌ Error on {native_split}: {e}")
            import traceback; traceback.print_exc()

        finally:
            if 'dataset' in locals():
                fo.delete_dataset(dataset.name)
            gc.collect()

if __name__ == "__main__":
    download_bags_original_labels()