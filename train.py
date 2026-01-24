from ultralytics import YOLO
from roboflow import Roboflow


def train_model():
    print("🚀 Downloading dataset from Roboflow...")

    rf = Roboflow(api_key="CgBMW2B8dFjhkDgmvKH2")
    project = rf.workspace("rjs-mwjgh").project("fieldscout")
    version = project.version(1)
    dataset = version.download("yolov8")

    print(f"✅ Data downloaded to: {dataset.location}")

    model = YOLO('yolov8n.pt')

    # 2. Train the model
    print("start training...")
    results = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=50,
        imgsz=640,
        plots=True
    )

    print("🎉 Training Complete!")


if __name__ == "__main__":
    train_model()