import os
import gc
import fiftyone as fo
import fiftyone.zoo as foz

# Optional custom storage locations
os.environ["FIFTYONE_DATASET_ZOO_DIR"] = "./fiftyone_zoo"
os.environ["FIFTYONE_DATABASE_DIR"] = "./fiftyone_db"

def download_bag_segmentation_dataset():
    # COCO/Sama-COCO class names are lowercase
    source_classes = ["backpack", "handbag", "suitcase"]
    export_classes = ["bag"]  # merged into one output class
    export_dir = "./coco_bag_seg"

    split_map = {
        "train": 700,
        "validation": 200
    }

    first_export = True

    for native_split, target_image_count in split_map.items():
        print(f"\n--- Processing {native_split} (target: {target_image_count} images) ---")

        dataset = None
        try:
            dataset = foz.load_zoo_dataset(
                "sama-coco",
                split=native_split,
                label_types=["segmentations"],
                classes=source_classes,
                only_matching=True,
                max_samples=target_image_count,
                shuffle=True,
                seed=42,
            )

            print(f"  Downloaded samples: {len(dataset)}")
            print(f"  Schema: {dataset.get_field_schema()}")

            # Sama-COCO stores labels in ground_truth
            view = dataset.filter_labels(
                "ground_truth",
                fo.ViewField("label").is_in(source_classes)
            )

            print(f"  Filtered samples: {len(view)}")

            # Merge all bag subclasses into one label: "bag"
            view = view.set_field(
                "ground_truth.detections.label",
                fo.ViewField("label").map_values({
                    "backpack": "bag",
                    "handbag": "bag",
                    "suitcase": "bag",
                })
            )

            # Skip exporting test split if no labels are present
            sample = view.first()
            has_labels = (
                sample is not None and
                hasattr(sample, "ground_truth") and
                sample.ground_truth is not None and
                len(sample.ground_truth.detections) > 0
            )

            if not has_labels:
                print(f"  Skipping {native_split}: no segmentation labels available")
                continue

            yolo_split = "val" if native_split == "validation" else native_split

            view.export(
                export_dir=export_dir,
                dataset_type=fo.types.YOLOv5Dataset,
                label_field="ground_truth",
                split=yolo_split,
                classes=export_classes,
                overwrite=first_export,
            )

            first_export = False
            print(f"✅ Exported {len(view)} images to '{yolo_split}'")

        except Exception as e:
            print(f"❌ Error on {native_split}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if dataset is not None:
                fo.delete_dataset(dataset.name)
            gc.collect()

if __name__ == "__main__":
    download_bag_segmentation_dataset()