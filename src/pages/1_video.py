import os
import json
import sys
from pathlib import Path
import cv2
import streamlit as st
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from deepsort_tracker import DeepSortTracker
from suspicious_bag_logic import SuspiciousBagAnalyzer

st.set_page_config(page_title="Upload And Run", layout="wide")

OUTPUT_DIR = "outputs"
UPLOAD_DIR = "uploads"
ROOT_DIR = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT_DIR / "runs" / "detect"

TRACKABLE_CLASSES = {"person", "bag", "handbag", "backpack", "suitcase"}
BAG_CLASSES = {"bag", "handbag", "backpack", "suitcase"}
BAG_STATUS_COLORS = {
    "normal": (0, 165, 255),
    "warming_up": (170, 170, 170),
    "unattended": (0, 255, 255),
    "abandoned": (0, 0, 255),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("Upload And Run")

# =========================
# Sidebar settings
# =========================
st.sidebar.header("Detection Settings")
best_weight_paths = sorted(RUNS_DIR.glob("**/weights/best.pt"))

if best_weight_paths:
    model_options = {}
    for weight_path in best_weight_paths:
        run_name = weight_path.parents[2].name
        stage_name = weight_path.parents[1].name
        display_name = f"{run_name} ({stage_name}) - best.pt"
        model_options[display_name] = str(weight_path)

    option_labels = list(model_options.keys())
    default_index = 0
    for idx, label in enumerate(option_labels):
        if "yolo11m" in label.lower():
            default_index = idx
            break

    selected_model_label = st.sidebar.selectbox("Model Weights", option_labels, index=default_index)
    selected_model_file = selected_model_label
    selected_model_path = model_options[selected_model_label]
else:
    selected_model_file = "best.pt"
    selected_model_path = st.sidebar.text_input(
        "Model Weights Path",
        value=str(RUNS_DIR / "weights" / "best.pt"),
    )

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.5, 0.05) #intersection over union,overlap of frames
distance_threshold = st.sidebar.slider("Bag-Person Distance Threshold (pixels)", 10, 500, 120, 10)
abandonment_time = st.sidebar.slider("Abandonment Time Threshold (seconds)", 1, 60, 10, 1)
min_bag_track_frames = st.sidebar.slider("Min Bag Track Frames", 1, 60, 12, 1)
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)
show_ids = st.sidebar.checkbox("Show Tracking IDs", value=True)


@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_path):
    if YOLO is None:
        return None
    return YOLO(weights_path)


yolo_model = None
model_error = None

if YOLO is None:
    model_error = "Ultralytics is not installed. Run: pip install ultralytics"
else:
    try:
        yolo_model = load_yolo_model(selected_model_path)
    except Exception as exc:
        model_error = f"Unable to load model '{selected_model_path}': {exc}"

if model_error:
    st.sidebar.warning(model_error)

if "deepsort_tracker" not in st.session_state:
    # Store tracker in session state so IDs can persist across Streamlit reruns.
    st.session_state.deepsort_tracker = DeepSortTracker()

deepsort_tracker = st.session_state.deepsort_tracker

if not deepsort_tracker.enabled:
    st.sidebar.info("DeepSORT disabled: install 'deep-sort-realtime' to enable ID tracking.")


def save_run_config(input_source: str, input_path: str):
    config = {
        "model_name": selected_model_file,
        "model_path": selected_model_path,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "distance_threshold": distance_threshold,
        "abandonment_time": abandonment_time,
        "min_bag_track_frames": min_bag_track_frames,
        "show_boxes": show_boxes,
        "show_ids": show_ids,
        "input_source": input_source,
        "input_path": input_path,
    }

    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
        json.dump(config, f, indent=4)


def build_detections(frame):
    if yolo_model is None:
        return []

    results = yolo_model.predict(
        source=frame,
        conf=confidence_threshold,
        iou=iou_threshold,
        verbose=False,
        imgsz=640,  # width, height
    )

    detections = []
    if not results:
        return detections

    result = results[0]
    names = result.names

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        class_name = str(names.get(cls_id, cls_id)).lower()
        if class_name not in TRACKABLE_CLASSES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float(box.conf[0].item()),
                "class_name": class_name,
            }
        )

    return detections


def detect_frame(frame, bag_analyzer=None):
    detections = build_detections(frame)
    annotated_frame = frame.copy()

    if deepsort_tracker.enabled:
        tracks = deepsort_tracker.update(detections, frame=frame)
        bag_status_by_id = {}
        events = []

        if bag_analyzer is not None:
            bag_status_by_id, events = bag_analyzer.update(tracks)

        for tracked_obj in tracks:
            x1, y1, x2, y2 = tracked_obj.bbox_xyxy
            class_name = tracked_obj.class_name.lower()
            track_id = tracked_obj.track_id

            if class_name == "person":
                color = (0, 200, 0)
            else:
                bag_state = bag_status_by_id.get(track_id, {})
                bag_status = bag_state.get("status", "normal")
                color = BAG_STATUS_COLORS.get(bag_status, (0, 165, 255))

            if show_boxes:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            if show_ids:
                label = f"{tracked_obj.class_name} #{tracked_obj.track_id}"
                if class_name in BAG_CLASSES:
                    bag_state = bag_status_by_id.get(track_id, {})
                    bag_status = bag_state.get("status")
                    if bag_status:
                        label = f"{label} [{bag_status}]"

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        return annotated_frame, tracks, events

    # Fallback visualization if DeepSORT is unavailable: draw plain detections.
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        class_name = det["class_name"].lower()
        color = (0, 200, 0) if class_name == "person" else (0, 165, 255)

        if show_boxes:
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(
            annotated_frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated_frame, detections, []


def process_uploaded_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"width:{width}")
    print(f"height:{height}")
    output_video_path = os.path.join(
        OUTPUT_DIR,
        f"detected_{Path(video_path).stem}.mp4",
    )

    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    bag_analyzer = SuspiciousBagAnalyzer(
        distance_threshold_px=distance_threshold,
        abandonment_time_sec=abandonment_time,
        fps=fps,
        min_bag_track_frames=min_bag_track_frames,
    )

    person_ids = set()
    bag_ids = set()
    alert_events = 0
    first_alert_timestamp = None
    all_events = []
    progress = st.progress(0)
    status_text = st.empty()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, objects, frame_events = detect_frame(frame, bag_analyzer=bag_analyzer)
        all_events.extend(frame_events)

        for event in frame_events:
            if event["status"] == "abandoned":
                alert_events += 1
                if first_alert_timestamp is None:
                    first_alert_timestamp = event["time"]

        for obj in objects:
            class_name = getattr(obj, "class_name", "").lower()
            track_id = getattr(obj, "track_id", None)

            if track_id is None:
                continue

            if class_name == "person":
                person_ids.add(track_id)
            elif class_name in BAG_CLASSES:
                bag_ids.add(track_id)

        writer.write(annotated_frame)

        frame_idx += 1
        if total_frames > 0:
            progress_value = min(frame_idx / total_frames, 1.0)
            progress.progress(progress_value)
            status_text.text(f"Processing... {int(progress_value * 100)}%")

    cap.release()
    writer.release()
    progress.empty()
    status_text.empty()

    events_csv_path = os.path.join(OUTPUT_DIR, "events.csv")
    with open(events_csv_path, "w") as f:
        f.write("time,bag_id,person_id,distance,status,previous_status\n")
        for event in all_events:
            f.write(
                f"{event['time']},{event['bag_id']},{event['person_id']},{event['distance']},{event['status']},{event['previous_status']}\n"
            )

    summary = {
        "total_people_detected": len(person_ids),
        "total_bags_detected": len(bag_ids),
        "alert_events": alert_events,
        "first_alert_timestamp": first_alert_timestamp,
        "status": "Completed",
        "output_video": output_video_path,
        "events_csv": events_csv_path,
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    return output_video_path, summary


# =========================
# Input mode selector
# =========================
mode = st.radio(
    "Select Input Mode",
    ["Upload Video", "Webcam Snapshot", "Live Webcam"],
    horizontal=True
)


if mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    #mp4 works, avi does not work
    if uploaded_file is not None:
        video_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"Uploaded: {uploaded_file.name}")
        st.success(f"video path: {video_path}")
        st.subheader("Original Video Preview")
        video_file = open(video_path,"rb")
        video_bytes = video_file.read()
        st.video(video_bytes)

        video_file.close()
        if st.button("▶ Run Detection on Video"):
            save_run_config("video_upload", video_path)
            if yolo_model is None:
                st.error("Model is not ready. Check the sidebar for setup errors.")
            else:
                if deepsort_tracker.enabled:
                    st.session_state.deepsort_tracker = DeepSortTracker()
                    deepsort_tracker = st.session_state.deepsort_tracker

                try:
                    output_video_path, summary = process_uploaded_video(video_path)
                    st.success("Detection completed.")
                    st.subheader("Detected Video")
                    with open(output_video_path, "rb") as f:
                        st.video(f.read())
                    st.json(summary)
                except Exception as exc:
                    st.error(f"Detection failed: {exc}")
    else:
        st.info("Upload a video to begin.")


elif mode == "Webcam Snapshot":
    st.write("Use your webcam to capture a single frame for testing.")
    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:
        image = Image.open(camera_image)
        image_path = os.path.join(UPLOAD_DIR, "webcam_snapshot.jpg")
        image.save(image_path)

        st.success("Snapshot captured successfully.")
        st.image(image, caption="Captured Snapshot", use_container_width=True)

        if st.button("▶ Run Detection on Snapshot"):
            save_run_config("webcam_snapshot", image_path)
            if yolo_model is None:
                st.error("Model is not ready. Check the sidebar for setup errors.")
            else:
                frame_bgr = cv2.imread(image_path)
                if frame_bgr is None:
                    st.error("Failed to load captured snapshot for detection.")
                else:
                    snapshot_analyzer = SuspiciousBagAnalyzer(
                        distance_threshold_px=distance_threshold,
                        abandonment_time_sec=abandonment_time,
                        fps=1,
                        min_bag_track_frames=1,
                    )
                    detected_frame, _, _ = detect_frame(frame_bgr, bag_analyzer=snapshot_analyzer)
                    detected_frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
                    st.image(detected_frame_rgb, caption="Snapshot Detection", use_container_width=True)
    else:
        st.info("Capture an image to begin.")


elif mode == "Live Webcam":
    st.write("Start your webcam for live preview. This is mainly for local testing.")
    run_camera = st.checkbox("Start Live Camera")

    frame_window = st.image([])
    info_box = st.empty()

    cap = None

    if run_camera:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Could not access webcam.")
        else:
            info_box.info("Webcam is running. Uncheck 'Start Live Camera' to stop.")
            live_bag_analyzer = SuspiciousBagAnalyzer(
                distance_threshold_px=distance_threshold,
                abandonment_time_sec=abandonment_time,
                fps=30,
                min_bag_track_frames=min_bag_track_frames,
            )

            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam.")
                    break

                # Apply YOLO + DeepSORT pipeline
                frame, _, _ = detect_frame(frame, bag_analyzer=live_bag_analyzer)

                # Convert BGR to RGB for Streamlit display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_rgb, channels="RGB", use_container_width=True)

                # Re-read checkbox state on rerun
                run_camera = st.session_state.get("Start Live Camera", True)

            cap.release()
    else:
        st.info("Tick 'Start Live Camera' to preview your webcam.")
