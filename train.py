from clearml import Dataset
from ultralytics import YOLO
from pathlib import Path

# 연결된 Dataset 가져오기 (첫 번째 Dataset)
ds = Dataset.get(dataset_project="vision_od", dataset_name="PPE")
data_root = Path(ds.get_local_copy())
data_yaml = data_root / "data.yaml"

model = YOLO("yolov8n.pt")
model.train(
    data=str(data_yaml),
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
)
