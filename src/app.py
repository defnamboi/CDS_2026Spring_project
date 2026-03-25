import streamlit as st
#streamlit run will auto look for pages/ folder, no need to import like node.js, will look for file in numeric order 
st.set_page_config(
    page_title="Abandoned Bag Detection System - D*** Scientist",
    layout="wide"
)

st.title("🧳 Abandoned Bag Detection System")
st.markdown("""
This app provides a simple surveillance interface for detecting **people**, **bags of all type**, and
potential **abandoned baggage events** from uploaded video footage.

### How to use
1. Go to **Upload And Run**
2. Upload a surveillance video
3. Configure thresholds
4. Run detection
5. View results, logs, and model info from the sidebar pages
""")


col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Detection Target", "People + Bags")

with col2:
    st.metric("Event States", "Normal / Unattended / Abandoned")

with col3:
    st.metric("Interface", "Streamlit UI")