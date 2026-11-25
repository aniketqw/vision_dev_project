"""Train the distortion tagger from scratch (YOLO11n architecture, no pretrained weights).

    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m visiondev.tagger.train --data outputs/tagger_dataset/data.yaml --device mps

Geometric augmentations are off: scaling, mosaic and translation resample pixels and would blur or alias
the very faults the model is learning. Flip and colour jitter stay (they do not change faults).
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def train(data: str, device: str = "mps", epochs: int = 40, batch: int = 128, project: str = "outputs/tagger_runs"):
    model = YOLO("yolo11n.yaml")
    model.train(
        data=data, imgsz=64, epochs=epochs, batch=batch, device=device, workers=4,
        project=project, name="tagger_v2", exist_ok=True, pretrained=False, amp=False, plots=False,
        patience=12, seed=0,
        mosaic=0.0, scale=0.0, translate=0.0, degrees=0.0, shear=0.0, perspective=0.0, mixup=0.0,
        flipud=0.0, fliplr=0.5, close_mosaic=0,
    )
    return Path(project) / "tagger_v2" / "weights" / "best.pt"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/tagger_dataset/data.yaml")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    print("BEST WEIGHTS:", train(a.data, a.device, a.epochs, a.batch))
