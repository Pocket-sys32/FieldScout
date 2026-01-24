import cv2
import os
import re
import easyocr
from datetime import datetime


reader = easyocr.Reader(['en'], gpu=False)


def extract_timestamp_from_video_ocr(video_path):
    """
    Reads the text burned into the video frame to find a date and time.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    # 1. Grab the first frame
    success, frame = cap.read()
    cap.release()

    if not success:
        return None, None

    # 2. OPTIMIZATION: Crop to bottom 20% and Top 10% for timestamps.
    # This makes it 5x faster and less confused by trees/grass.
    height, width, _ = frame.shape

    # Bottom strip (most common for trail cams)
    bottom_crop = frame[int(height * 0.8):height, 0:width]

    # Top strip (sometimes used) (not in our case but good to have for future cameras)
    top_crop = frame[0:int(height * 0.1), 0:width]

    # 3. Run OCR (Read Text)
    # We combine results from top and bottom
    print(f"👀 Scanning video frame for text...")
    results_bottom = reader.readtext(bottom_crop, detail=0)
    results_top = reader.readtext(top_crop, detail=0)
    all_text = " ".join(results_bottom + results_top)

    print(f"   -> Found text: {all_text}")

    # 4. Regex Magic (Find patterns like YYYY-MM-DD or MM/DD/YYYY)
    date_match = re.search(r'(\d{4}[-/]\d{2}[-/]\d{2})|(\d{2}[-/]\d{2}[-/]\d{4})', all_text)

    # Time Pattern: 14:30:05
    time_match = re.search(r'\d{2}:\d{2}:\d{2}', all_text)

    date_str = date_match.group(0) if date_match else None
    time_str = time_match.group(0) if time_match else None

    # Normalize Date (Optional: Force YYYY-MM-DD if needed)
    return date_str, time_str


def get_video_timestamp(video_path):
    """
    Master function: Try OCR first, then Metadata, then File System.
    """
    # STRATEGY 1: OCR (Visual Vision) - Accurate for trail cameras
    try:
        ocr_date, ocr_time = extract_timestamp_from_video_ocr(video_path)
        if ocr_date and ocr_time:
            print(f"✅ OCR Success: {ocr_date} {ocr_time}")
            return ocr_date, ocr_time
    except Exception as e:
        print(f"⚠️ OCR Failed: {e}")

    # STRATEGY 2: File Metadata (Fallback)
    try:
        timestamp = os.path.getmtime(video_path)
        dt_obj = datetime.fromtimestamp(timestamp)
        return dt_obj.strftime("%Y-%m-%d"), dt_obj.strftime("%H:%M:%S")
    except:
        now = datetime.now()
        return now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")


# Keeps the ability to create more training data if needed later.
def extract_frames_from_video(video_path, output_folder, interval_seconds=2):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error: Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    count = 0
    saved_count = 0

    print(f"✂️  Processing {video_name}...")

    while True:
        success, frame = cap.read()
        if not success:
            break

        if count % frame_interval == 0:
            output_filename = f"{output_folder}/{video_name}_frame_{saved_count}.jpg"
            cv2.imwrite(output_filename, frame)
            saved_count += 1

        count += 1

    cap.release()
    print(f"✅ Saved {saved_count} images from {video_name}")


if __name__ == "__main__":
    # If run directly, assume we want to chop videos for training
    # This block won't run when imported by app/ui.py
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_script_path))
    raw_folder = os.path.join(project_root, "data", "raw")
    output_folder = os.path.join(project_root, "data", "training_images")

    if os.path.exists(raw_folder):
        files = [f for f in os.listdir(raw_folder) if f.lower().endswith(('.mp4', '.avi'))]
        for f in files:
            extract_frames_from_video(os.path.join(raw_folder, f), output_folder)


def process_all_videos():
    # Robust path finding
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_script_path))

    raw_folder = os.path.join(project_root, "data", "raw")
    output_folder = os.path.join(project_root, "data", "training_images")

    if not os.path.exists(raw_folder):
        print(f"❌ ERROR: Could not find folder: {raw_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    # Case-insensitive search
    all_files = os.listdir(raw_folder)
    video_files = [f for f in all_files if f.lower().endswith(('.mp4', '.avi', '.mov', '.m4v'))]

    if not video_files:
        print(f"⚠️ No videos found in {raw_folder}")
        return

    print(f"Found {len(video_files)} videos. Starting extraction...")

    for video in video_files:
        full_path = os.path.join(raw_folder, video)
        extract_frames_from_video(full_path, output_folder)


if __name__ == "__main__":
    process_all_videos()