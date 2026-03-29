import os
import json
import time
import cv2
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Upload And Run", layout="wide")

OUTPUT_DIR = "outputs"
UPLOAD_DIR = "uploads"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("Upload And Run")

# =========================
# Sidebar settings
# =========================
st.sidebar.header("Detection Settings")
model_name = st.sidebar.selectbox("Model", ["YOLOv8", "SSD"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.5, 0.05) #intersection over union,overlap of frames
distance_threshold = st.sidebar.slider("Bag-Person Distance Threshold (pixels)", 10, 500, 120, 10)
abandonment_time = st.sidebar.slider("Abandonment Time Threshold (seconds)", 1, 60, 10, 1)
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)
show_ids = st.sidebar.checkbox("Show Tracking IDs", value=True)


def save_run_config(input_source: str, input_path: str):
    config = {
        "model_name": model_name,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "distance_threshold": distance_threshold,
        "abandonment_time": abandonment_time,
        "show_boxes": show_boxes,
        "show_ids": show_ids,
        "input_source": input_source,
        "input_path": input_path,
    }

    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
        json.dump(config, f, indent=4)


def create_dummy_outputs():
    summary = {
        "total_people_detected": 12,
        "total_bags_detected": 5,
        "alert_events": 2,
        "first_alert_timestamp": "00:18",
        "processing_time_sec": 4.8,
        "status": "Completed"
    }

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    events_csv = os.path.join(OUTPUT_DIR, "events.csv")
    with open(events_csv, "w") as f:
        f.write("time,bag_id,person_id,distance,status\n")
        f.write("00:12,Bag_1,Person_2,45,normal\n")
        f.write("00:18,Bag_1,None,170,unattended\n")
        f.write("00:25,Bag_1,None,220,abandoned\n")


def simulate_processing():
    st.info("Running analysis...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(100):
        time.sleep(0.02)
        progress_bar.progress(i + 1)
        status_text.text(f"Processing... {i + 1}%")

    create_dummy_outputs()
    st.success("Detection completed.")
    st.warning("This page is currently using simulated outputs. Replace this with your real detection pipeline later.")
    st.markdown("### Next Step")
    st.write("Go to the **Results** and **Event Log** pages from the sidebar.")


# Optional placeholder for your real detection function
def detect_frame(frame):
    """
    Replace this with your real detection code.
    Input: OpenCV frame (BGR or RGB depending on your pipeline)
    Output: annotated frame
    """
    return frame


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
            simulate_processing()
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
            simulate_processing()
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

            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam.")
                    break

                # Apply detection placeholder
                frame = detect_frame(frame)

                # Convert BGR to RGB for Streamlit display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_rgb, channels="RGB", use_container_width=True)

                # Re-read checkbox state on rerun
                run_camera = st.session_state.get("Start Live Camera", True)

            cap.release()
    else:
        st.info("Tick 'Start Live Camera' to preview your webcam.")

    st.markdown("### Optional")
    st.write("For most project demos, uploaded video is more stable than live webcam.")