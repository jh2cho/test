from clearml import Dataset
from ultralytics import YOLO
from pathlib import Path
import yaml, os

DATASET_ID = "57a033aab55141b0ae379b0b31465f9b" 

# 1) Dataset 내려받기
data_root = Path(Dataset.get(dataset_id=DATASET_ID).get_local_copy()).resolve()
orig_yaml = data_root / "data.yaml"

print("CWD:", os.getcwd())
print("Dataset root:", data_root)
print("Original yaml:", orig_yaml)

cfg = yaml.safe_load(orig_yaml.read_text())

# 2) path를 dataset root 절대경로로 강제
cfg["path"] = str(data_root)

# 3) train/val/test는 상대경로 유지(권장)
#    (이미 images/train 형태면 그대로 두면 됨)
fixed_yaml = Path("data_fixed.yaml").resolve()
fixed_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))

print("Fixed yaml:", fixed_yaml)
print("Fixed yaml content:\n", fixed_yaml.read_text())

# 4) 실제 경로 존재 체크 (여기서 터지면 폴더 구조 문제)
for k in ("train", "val", "test"):
    if k in cfg:
        p = (data_root / cfg[k]).resolve() if not str(cfg[k]).startswith("/") else Path(cfg[k])
        print(k, "->", p, "exists:", p.exists())

# 5) YOLO 학습
model = YOLO("yolov8n.pt")
model.train(
    data=str(fixed_yaml),
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
)
