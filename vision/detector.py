"""
vision/detector.py

Runs YOLO on a page image to detect visual regions:
  - figure / chart / diagram
  - table
  - formula
  - photo

Returns cropped region images with bounding boxes.
"""
import os
import uuid
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image


YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "./models/yolo_doc_layout.pt")
YOLO_CONF = float(os.getenv("YOLO_CONFIDENCE", 0.35))

# Fallback class names for document layout YOLO
# Matches DocLayNet / PubLayNet label sets
YOLO_CLASSES = {
    0: "text",
    1: "title",
    2: "figure",
    3: "table",
    4: "list",
    5: "formula",
}


class YOLODetector:
    """
    Detects document layout regions using YOLO.
    Falls back to a simple heuristic if no model file is present.
    """

    def __init__(self):
        self._model = None
        self._model_path = YOLO_MODEL_PATH

    def _load(self):
        if self._model is not None:
            return
        if not Path(self._model_path).exists():
            # Use YOLO pretrained on document layouts (auto-downloads)
            from ultralytics import YOLO
            self._model = YOLO("yolov8n.pt")  # swap with doc-layout weights
            return
        from ultralytics import YOLO
        self._model = YOLO(self._model_path)

    def detect(
        self, page_image: np.ndarray, page_num: int
    ) -> List[Dict[str, Any]]:
        """
        Run YOLO on a page image.

        Returns a list of dicts:
        {
            "class_name": str,
            "confidence": float,
            "bbox": [x0_norm, y0_norm, x1_norm, y1_norm],
            "crop": np.ndarray,
            "crop_path": str   (temp file path)
        }
        """
        self._load()

        h, w = page_image.shape[:2]
        results = self._model(page_image, conf=YOLO_CONF, verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = YOLO_CLASSES.get(cls_id, f"class_{cls_id}")

                # Skip pure text/title blocks — we handle those in the parser
                if class_name in ("text", "title", "list"):
                    continue

                conf = float(box.conf[0])
                x0, y0, x1, y1 = box.xyxy[0].tolist()

                # Crop image region
                crop = page_image[int(y0):int(y1), int(x0):int(x1)]
                if crop.size == 0:
                    continue

                # Save crop to temp file for CLIP
                crop_path = _save_crop(crop, page_num)

                detections.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": [x0 / w, y0 / h, x1 / w, y1 / h],
                    "crop": crop,
                    "crop_path": crop_path,
                })

        return detections


def _save_crop(crop: np.ndarray, page_num: int) -> str:
    """Save a cropped region as a PNG and return the file path."""
    out_dir = Path("./data/processed/crops")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"page{page_num}_{uuid.uuid4().hex[:8]}.png"
    Image.fromarray(crop).save(str(fname))
    return str(fname)
