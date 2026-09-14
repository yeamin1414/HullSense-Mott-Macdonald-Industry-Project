"""Optional YOLO object-detection backend.

The original concept design calls for "a self-trained YOLO model [that] spots
the stains and splits the hull into sections." A real YOLO model needs a
labelled dataset of actual hull photos, which this repo does not have — so
this module is the honest, documented path to that upgrade rather than a
fake demo. It is not required for the system to run: `detector.py` only uses
this backend if `ultralytics` is installed AND a trained weights file is
supplied.

To actually train one, once real photos + Roboflow/LabelImg annotations of
stains exist:

    pip install ultralytics
    from src.vision.yolo_detector import train_yolo
    train_yolo("path/to/data.yaml", epochs=100)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:  # pragma: no cover
    ULTRALYTICS_AVAILABLE = False


class YOLOStainDetector:
    """Wraps a trained Ultralytics YOLO model and converts its bounding-box
    predictions into the same (rows, cols) dirtiness grid every other
    detector in this repo produces, so it's a drop-in replacement."""

    def __init__(self, weights_path: str | Path, rows: int = 4, cols: int = 6, conf: float = 0.25):
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("ultralytics is not installed — run `pip install ultralytics`.")
        self.model = YOLO(str(weights_path))
        self.rows, self.cols = rows, cols
        self.conf = conf

    def predict_grid(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        results = self.model.predict(image, conf=self.conf, verbose=False)
        grid = np.zeros((self.rows, self.cols), dtype=np.float32)
        cell_h, cell_w = h / self.rows, w / self.cols
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            r = min(self.rows - 1, int(cy // cell_h))
            c = min(self.cols - 1, int(cx // cell_w))
            grid[r, c] = max(grid[r, c], float(box.conf[0]))
        return grid


def train_yolo(data_yaml: str, epochs: int = 100, imgsz: int = 640, model_size: str = "yolov8n.pt") -> str:
    """Fine-tunes a YOLOv8 nano model on a labelled stain dataset.

    `data_yaml` follows the standard Ultralytics dataset spec (train/val
    image dirs + class names). Returns the path to the best trained weights.
    """
    if not ULTRALYTICS_AVAILABLE:
        raise RuntimeError("ultralytics is not installed — run `pip install ultralytics`.")
    model = YOLO(model_size)
    results = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)
    return str(Path(results.save_dir) / "weights" / "best.pt")
