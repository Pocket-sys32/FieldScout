import os
from ultralytics import YOLO


def run_visual_test():
    # 1. Load your trained model
    model_path = 'models/best.pt'
    if not os.path.exists(model_path):
        print("❌ Error: Could not find models/best.pt. Did you move the file?")
        return

    print(f"🌲 Loading FieldScout Model: {model_path}...")
    model = YOLO(model_path)

    # 2. Find the video in data/raw
    raw_folder = 'data/raw'
    video_files = [f for f in os.listdir(raw_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    if not video_files:
        print("❌ No videos found in data/raw!")
        return

    # Pick the first video found
    test_video = os.path.join(raw_folder, video_files[0])
    print(f"🎥 Analyzing Video: {test_video}")

    # 3. Run Prediction & Save Output
    # save=True tells YOLO to write a new video file with boxes drawn
    # conf=0.5 means "Only show detection if you are 50% sure"
    # Change conf=0.5 to conf=0.1 (10% confidence)
    results = model.predict(source=test_video, save=True, conf=0.01)

    print("-" * 30)
    print("✅ Video processing complete!")
    print(f"📂 Output saved to: {results[0].save_dir}")
    print("Go open that folder and watch your video!")


if __name__ == "__main__":
    run_visual_test()