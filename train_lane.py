from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # small model

model.train(
    data="lane_dataset/data.yaml",
    epochs=10,
    imgsz=640,
    batch=4,
    name="lane_model"
)