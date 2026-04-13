# Abandoned Bag Detection System

Our project is a computer vision system that automatically detects and flags **unattended or abandoned bags** in public surveillance footage.

---
## Problem Statement

Current surveillance measures for detecting suspicious activity such as unattended bag abandonment, rely heavily on manual monitoring and human intervention. This approach has several critical limitations:

- **Resource-intensive:** Requires continuous staffing across multiple locations simultaneously  

- **Inconsistent:** Human judgment and attentiveness vary significantly between operators and over time  

- **Contextually demanding:** Identifying abandonment requires persistent observation and an understanding of temporal behaviour, which is difficult to maintain in busy, high-traffic environments like MRT stations  

This project addresses these limitations by proposing a computer vision-based system that automates detection and classification of suspicious bag abandonment — providing a more consistent, scalable complement to existing security measures.

---

## Our Solution

Our project aims to tackle this proble by using a system automates the detection of suspicious bag abandonment in public spaces (e.g. MRT stations) to complement existing measures by combining:

- **YOLO11m** — object detection to identify people and bags in each frame
- **DeepSORT** — multi-object tracking to maintain persistent IDs across frames
- **Finite State Machine (FSM)** — behavioural logic to classify each bag's state over time
- **Streamlit** — a web UI for uploading footage, tuning parameters, and reviewing results

---

##  Repository Structure

```
CDS_2026Spring_project/
├── data/           # Processed COCO dataset (images + YOLO annotations)
├── notebook/       # Jupyter notebooks for EDA, preprocessing, training, evaluation
├── runs/           # YOLO training run outputs (weights, metrics, logs)
├── src/            # Core Python source (detection, tracking, FSM logic, Streamlit app)
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- Python added to your system `PATH`
- GPU recommended for real-time inference

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/defnamboi/CDS_2026Spring_project.git
   cd CDS_2026Spring_project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FiftyOne** (required for dataset management and visualisation):
   ```bash
   python -m pip install fiftyone
   ```

### Running the App

```bash
streamlit run src/app.py
```

Then in the UI:
1. Go to the **Video** page
2. Upload a surveillance video (`.mp4`, `.avi`, `.mkv`, `.mpeg4`, max 200MB)
3. Adjust detection parameters in the sidebar
4. Click **Run Detection on Video**
5. View the annotated output video, results summary, and event logs

---

##  Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv11m (Ultralytics) |
| Object Tracking | DeepSORT |
| Dataset Management | FiftyOne |
| Web Interface | Streamlit |
| Language | Python |
| Notebooks | Jupyter |
| Training Data | COCO (via FiftyOne API) |

---

##  Detection Parameters

These can be tuned directly in the Streamlit sidebar without changing any code:

| Parameter | Description |
|---|---|
| **Model Weights** | Select which trained YOLO weights to use |
| **Confidence Threshold** | Minimum detection confidence (default: 0.50) |
| **IoU Threshold** | Intersection-over-Union threshold for NMS (default: 0.25) |
| **Bag-Person Distance Threshold (px)** | Max pixel distance for a bag to be considered "near" its owner (default: 100) |
| **Abandonment Time Threshold (s)** | Seconds unattended before a bag is flagged as abandoned (default: 3) |
| **Min Bag Track Frames** | Minimum frames a bag must be tracked before state logic applies |
| **Show Bounding Boxes** | Toggle bounding box overlay |
| **Show Tracking IDs** | Toggle DeepSORT ID labels |

---

##  How It Works

### 1. Object Detection — YOLO11m

The system uses a **YOLO11m** model trained on a curated COCO subset (5,049 images) to detect two classes: `person` and `bag`. YOLO was selected over Faster R-CNN and SSD based on benchmarking results:

| Model | Precision | Recall | mAP@50 | Inference Speed (ms) |
|---|---|---|---|---|
| **YOLO11m** | **0.847** | **0.743** | **0.827** | **18.7** |
| Faster R-CNN | 0.0839 | 0.5124 | 0.6188 | 62.77 |
| SSD | 0.7694 | 0.6158 | 0.3294 | 282.19 |

### 2. Multi-Object Tracking — DeepSORT

DeepSORT assigns persistent IDs to each detected person and bag across video frames. Key parameters:

| Parameter | Value | Purpose |
|---|---|---|
| `max_age` | 100 | Frames to retain a track without detection (handles occlusion) |
| `n_init` | 10 | Frames required before confirming a new track |
| `max_iou_distance` | 0.7 | IoU gate for spatial matching |
| `max_cosine_distance` | 0.45 | Appearance matching tolerance |

### 3. Finite State Machine (FSM)

Each tracked bag is independently assigned one of four states:

| State | Description |
|---|---|
| `warming_up` | Bag is newly detected; too early to judge |
| `normal` | Bag is near its owner or being carried |
| `unattended` | Bag has been separated from owner beyond the grace period |
| `abandoned` | Bag has been unattended past the abandonment threshold (terminal state) |

State transitions are governed by proximity to the nearest person, movement detection, configurable time thresholds, and a spatial memory system that prevents duplicate alerts for the same location.

---

##  Evaluation Results

### DeepSORT Tracker — Activity Detection

Tested against footage filmed on campus across multiple locations and scenarios:

| | Predicted: Suspicious | Predicted: Non-Suspicious |
|---|---|---|
| **Actual: Suspicious** | 8 | 1 |
| **Actual: Non-Suspicious** | 0 | 3 |

**Overall Accuracy: 91.7%**

### DeepSORT Tracker — Object Identification

Average across all suspicious test footages:

| Metric | Score |
|---|---|
| Precision | 0.74 |
| Recall | 0.83 |

---

##  Dataset

The final model was trained on a **COCO-derived dataset** (5,049 images, classes: `person` and `bag`) downloaded using the FiftyOne API. Two other datasets were evaluated and excluded:

- **Roboflow Abandoned Object Dataset** — excluded due to label incompatibility and domain gap (CCTV-only angles)
- **Open Images v7** — excluded due to poor handling of person-carrying-bag interactions

### Data Preprocessing Pipeline

1. **Image integrity checking** — removes corrupted, empty, or malformed images
2. **YOLO label validation** — validates annotation format, class IDs, and bounding box bounds
3. **Annotation standardisation** — remaps all bag-type labels (`handbag`, `backpack`, `suitcase`) to a single `bag` class
4. **Spatial resolution standardisation** — all images resized to 640×640px
5. **Min-Max normalisation** — pixel intensities normalised to [0, 1]
6. **Dataset split** — 70% train / 20% validation / 10% test

