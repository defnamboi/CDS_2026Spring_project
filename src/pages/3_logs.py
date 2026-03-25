import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Event Log", layout="wide")

OUTPUT_DIR = "outputs"
events_path = os.path.join(OUTPUT_DIR, "events.csv")


if os.path.exists(events_path):
    df = pd.read_csv(events_path)

    st.subheader("Detected Events")
    st.dataframe(df, use_container_width=True)

    st.subheader("Filter by Status")
    selected_status = st.multiselect(
        "Choose status",
        options=df["status"].unique().tolist(),
        default=df["status"].unique().tolist()
    )

    filtered_df = df[df["status"].isin(selected_status)]
    st.dataframe(filtered_df, use_container_width=True)

    st.subheader("Quick Summary")
    st.write(filtered_df["status"].value_counts())

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Event Log CSV",
        data=csv_data,
        file_name="filtered_event_log.csv",
        mime="text/csv"
    )
else:
    st.warning("No event log found yet. Run detection first.")