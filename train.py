from clearml import Task, Dataset
from ultralytics import YOLO
from pathlib import Path
import yaml


DEFAULTS = {
    "dataset_id": "57a033aab55141b0ae379b0b31465f9b",  
    "model": "yolov8n.pt",
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "device": 0,
}

task = Task.current_task()

params = task.connect(DEFAULTS, name="yolo")

data_root = Path(Dataset.get(dataset_id=params["dataset_id"]).get_local_copy()).resolve()
orig_yaml = data_root / "data.yaml"
cfg = yaml.safe_load(orig_yaml.read_text())

cfg["path"] = str(data_root)
fixed_yaml = Path("data_fixed.yaml").resolve()
fixed_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))

print("Dataset root:", data_root)
print("Using yaml:", fixed_yaml)

model = YOLO(params["model"])
results = model.train(
    data=str(fixed_yaml),
    epochs=int(params["epochs"]),
    imgsz=int(params["imgsz"]),
    batch=int(params["batch"]),
    device=int(params["device"]),
)

save_dir = Path(model.trainer.save_dir).resolve()
print("Save dir:", save_dir)
