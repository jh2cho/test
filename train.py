from clearml import Dataset
from ultralytics import YOLO
from pathlib import Path

DATASET_ID = "ab69a5f04c4f4ba1b4ad12f9e15fbc88"

# ClearML UI에서 "Add Task.init call"을 켰다면 여기서 Task.init() 호출하지 말 것!
# (UI가 자동으로 Task.init을 삽입해줌)

# Dataset 다운로드
data_root = Path(Dataset.get(dataset_id=DATASET_ID).get_local_copy())
data_yaml = data_root / "data.yaml"

print("Dataset local path:", data_root)
print("Using data.yaml:", data_yaml)

# YOLO 학습
model = YOLO("yolov8n.pt")
model.train(
    data=str(data_yaml),
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
)
