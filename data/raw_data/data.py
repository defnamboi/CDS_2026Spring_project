import fiftyone as fo
import fiftyone.zoo as foz
print("hello world")
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=["Backpack", "Suitcase"],
    max_samples=500
)

session = fo.launch_app(dataset)

for sample in dataset:
    print(sample.filepath)