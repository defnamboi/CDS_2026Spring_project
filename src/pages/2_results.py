import os
import json
import streamlit as st

st.set_page_config(page_title="Results", layout="wide")

OUTPUT_DIR = "outputs"
summary_path = os.path.join(OUTPUT_DIR, "summary.json")
video_path = os.path.join(OUTPUT_DIR, "processed_video.mp4")

st.title("📊 Detection Results")

if os.path.exists(summary_path):
    with open(summary_path, "r") as f:
        summary = json.load(f)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("People Detected", summary.get("total_people_detected", 0))
        st.metric("Bags Detected", summary.get("total_bags_detected", 0))

    with col2:
        st.metric("Alert Events", summary.get("alert_events", 0))
        st.metric("First Alert", summary.get("first_alert_timestamp", "N/A"))

    with col3:
        st.metric("Processing Time (s)", summary.get("processing_time_sec", 0))
        st.metric("Status", summary.get("status", "Unknown"))

    if summary.get("alert_events", 0) > 0:
        st.error(f"Alert: {summary['alert_events']} suspicious baggage event(s) detected.")
    else:
        st.success("No suspicious baggage detected.")

else:
    st.warning("No summary available yet. Run detection first.")

st.subheader("Processed Video")

if os.path.exists(video_path):
    st.video(video_path)
else:
    st.info("Processed video not found yet. Your backend should save it as outputs/processed_video.mp4")
