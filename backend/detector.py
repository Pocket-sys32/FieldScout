import cv2
from ultralytics import YOLO


class AnimalDetector:
    def __init__(self, model_path=None):
        # No model (self-made model)
        if model_path is None:
            model_path = 'models/best.pt'

        print(f"🌲 Loading FieldScout Model: {model_path}")
        self.model = YOLO(model_path)

    def process_video(self, video_path):
        """
        Analyzes a video and returns the SINGLE best detection found.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        skip_frames = int(fps)

        best_detection = None
        max_conf = 0.0
        frame_count = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_count % skip_frames == 0:
                # 1. Low confidence to catch everything (data is sparse at first)
                results = self.model(frame, verbose=False, conf=0.01)

                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0])

                        # 2. Update best detection if this one is clearer
                        if conf > max_conf and conf > 0.01:
                            max_conf = conf

                            # 3. DIRECT ASSIGNMENT (No variables to mess up)
                            best_detection = {
                                "species": self.model.names[int(box.cls[0])],
                                "confidence": conf
                            }

            frame_count += 1

        cap.release()
        return best_detection