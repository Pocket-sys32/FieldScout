import streamlit as st
import os
import sys
import tempfile
import pandas as pd
import time

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from backend.detector import AnimalDetector
from backend.sheets import SheetLogger
from utils.video_utils import get_video_timestamp

# --- CONFIGURATION ---
CREDENTIALS_FILE = os.path.join(parent_dir, "credentials.json")
SHEET_NAME = "Winter 26 Game Cam Data"
RAW_FOLDER = os.path.join(parent_dir, "data", "raw")

st.set_page_config(page_title="FieldScout", page_icon="🌲", layout="wide")
st.title("🌲 FieldScout: Trail Camera Animal ID")



@st.cache_resource
def load_system():
    model_path = os.path.join(parent_dir, "models", "best.pt")
    detector = AnimalDetector(model_path)
    logger = SheetLogger(CREDENTIALS_FILE, SHEET_NAME)
    return detector, logger


try:
    detector, logger = load_system()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["👁️ Single Inspector", "🚀 Batch Processor", "📊 Data Logs"])

# ==========================================
# TAB 1: SINGLE INSPECTOR
# ==========================================
with tab1:
    st.header("Inspect a Single Video")
    st.info(
        "⚠️ Note: Use batch-processor for folders of data! - RJ")
    uploaded_file = st.file_uploader("Drop a video here", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[1]
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        col1, col2 = st.columns(2)
        with col1:
            if file_ext.lower() == '.avi':
                st.info("ℹ️ .AVI video hidden (Browser incompatible)")
            else:
                st.video(video_path)

        with col2:
            if st.button("Analyze This Video", type="primary"):
                # 1. OCR SCANNING
                with st.spinner("👀 Reading video timestamp..."):
                    # We call our new smart function here
                    real_date, real_time = get_video_timestamp(video_path)

                # 2. AI DETECTION
                with st.spinner("🧠 Identifying animal..."):
                    result = detector.process_video(video_path)

                    if result:
                        st.success(f"**{result['species']}** detected!")
                        st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")

                        # Show what we found
                        if real_date:
                            st.caption(f"📅 Timestamp found in video: **{real_date} at {real_time}**")
                        else:
                            st.caption("⚠️ Could not read timestamp visually. Using upload time.")

                        if logger.sheet:
                            logger.log_detection(
                                result['species'],
                                result['confidence'],
                                uploaded_file.name,
                                date_str=real_date,
                                time_str=real_time
                            )
                            st.toast(f"Saved! ({real_date})")
                    else:
                        st.warning("No animals found.")
# ==========================================
# TAB 2: BATCH PROCESSOR
# ==========================================
with tab2:
    st.header("Batch Process Folder")

    try:
        files = [f for f in os.listdir(RAW_FOLDER) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        st.info(f"📂 Found **{len(files)}** videos in {RAW_FOLDER}")
    except FileNotFoundError:
        st.error(f"❌ Could not find folder: {RAW_FOLDER}")
        files = []

    if files and st.button(f"🚀 Process All {len(files)} Videos"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_count = 0

        for i, filename in enumerate(files):
            status_text.text(f"Analyzing {filename}...")
            progress_bar.progress((i + 1) / len(files))

            video_path = os.path.join(RAW_FOLDER, filename)

            # 1. Run AI
            result = detector.process_video(video_path)

            if result:
                # 2. Getting meta-data of files
                # This reads the file creation date from storage
                d_str, t_str = get_video_timestamp(video_path)

                # 3. Log using that specific date/time
                if logger.sheet:
                    logger.log_detection(
                        result['species'],
                        result['confidence'],
                        filename,
                        date_str=d_str,  # <--- Date
                        time_str=t_str  # <--- Time
                    )
                    log_count += 1

            time.sleep(0.1)

        st.success(f"🎉 Processed {len(files)} videos. Found animals in {log_count} of them.")

# ==========================================
# TAB 3: DATA LOGS (Google Sheet Int.)
# ==========================================
with tab3:
    st.header("Database View")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()

    if logger.sheet:
        try:
            raw_data = logger.sheet.get_all_values()

            if len(raw_data) > 1:
                all_headers = raw_data[0]
                all_rows = raw_data[1:]
                valid_indices = [i for i, header in enumerate(all_headers) if header.strip() != ""]

                if valid_indices:
                    clean_headers = [all_headers[i] for i in valid_indices]
                    clean_rows = []
                    for row in all_rows:
                        cleaned_row = [row[i] if i < len(row) else "" for i in valid_indices]
                        clean_rows.append(cleaned_row)

                    df = pd.DataFrame(clean_rows, columns=clean_headers)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No headers found.")
            else:
                st.info("Sheet is empty.")

        except Exception as e:
            st.error(f"Error reading sheet: {e}")