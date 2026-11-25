"""Run the trained distortion tagger on images."""
from pathlib import Path
from typing import Dict, List

import numpy as np
from ultralytics import YOLO

from ..config import DEFAULT_TAGGER


class Tagger:
    """Wraps the YOLO tagger. Input images are RGB uint8 HxWx3 arrays (CIFAR-sized).

    tags(img)     -> {"blur": 0.97, "noise": 0.31}   (empty dict = no fault found)
    top_tag(img)  -> ("blur", 0.97) or ("none", None)

    conf=0.25 is the threshold used in every reported result. There is no forced guess:
    if nothing clears the threshold the answer is "none", which is a real outcome
    (96% of natural CIFAR failures end up here).
    """

    def __init__(self, weights: str = str(DEFAULT_TAGGER), imgsz: int = 64, conf: float = 0.25, device: str = "cpu"):
        self.model = YOLO(str(weights))
        self.names = self.model.model.names
        self.imgsz, self.conf, self.device = imgsz, conf, device

    def tags_batch(self, images_rgb: List[np.ndarray], batch_size: int = 128) -> List[Dict[str, float]]:
        out = []
        for i in range(0, len(images_rgb), batch_size):
            chunk = [im[:, :, ::-1].copy() for im in images_rgb[i:i + batch_size]]  # ultralytics expects BGR arrays
            for r in self.model.predict(chunk, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False):
                d: Dict[str, float] = {}
                for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    name = self.names[int(c)]
                    d[name] = max(d.get(name, 0.0), float(s))
                out.append(d)
        return out

    def tags(self, image_rgb: np.ndarray) -> Dict[str, float]:
        return self.tags_batch([image_rgb])[0]

    def top_tag(self, image_rgb: np.ndarray):
        d = self.tags(image_rgb)
        if not d:
            return "none", None
        name = max(d, key=d.get)
        return name, d[name]
