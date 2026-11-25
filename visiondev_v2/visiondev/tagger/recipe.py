"""Fault-injection recipe v2 for 32x32 images (fixes and extends the original DIV2K notebook recipe).

Changes vs the notebook:
  * noise: signed arithmetic + clip (the notebook's astype(np.uint8) wrapped negative noise to ~200+ and
    saturated half the patch to white)
  * injected at the SAME resolution it is used at (32x32), severities from mild to strong
  * 25% clean images (empty label file -> YOLO background), 2-3 faults on one image, global or local faults
Class ids are unchanged: 0 blur, 1 noise, 2 pixelate, 3 jpeg.
"""
import cv2
import numpy as np

from ..config import TAGS as FAULTS  # class ids: 0 blur, 1 noise, 2 pixelate, 3 jpeg
ORDER = ["pixelate", "blur", "noise", "jpeg"]  # canonical application order when faults are stacked


def apply_fault(roi, kind, rng):
    """roi: HxWx3 uint8 (RGB). Returns a distorted copy."""
    if kind == "blur":
        sigma = rng.uniform(0.6, 2.5)
        k = max(3, int(2 * round(3 * sigma) + 1))
        return cv2.GaussianBlur(roi, (k, k), sigma)
    if kind == "noise":
        s = rng.uniform(8, 50)  # std on the 0-255 scale
        return np.clip(roi.astype(np.float32) + rng.normal(0, s, roi.shape), 0, 255).astype(np.uint8)
    if kind == "pixelate":
        f = int(rng.integers(2, 5))
        h, w = roi.shape[:2]
        small = cv2.resize(roi, (max(2, w // f), max(2, h // f)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    if kind == "jpeg":
        q = int(rng.integers(8, 71))
        _, enc = cv2.imencode(".jpg", cv2.cvtColor(roi, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB)
    raise ValueError(kind)


def make_sample(img, rng, p_clean=0.25):
    """img: 32x32x3 uint8 RGB. Returns (distorted_img, labels) with labels = [(cls, cx, cy, w, h)] normalised."""
    H, W = img.shape[:2]
    out = img.copy()
    if rng.random() < p_clean:
        return out, []
    n = int(rng.choice([1, 2, 3], p=[0.45 / 0.75, 0.25 / 0.75, 0.05 / 0.75]))
    chosen = {str(k) for k in rng.choice(FAULTS, size=n, replace=False)}  # draw ONCE per image
    kinds = [k for k in ORDER if k in chosen]
    labels = []
    for kind in kinds:
        if rng.random() < 0.5:  # global fault
            x0, y0, x1, y1 = 0, 0, W, H
        else:  # local patch
            bw, bh = int(rng.integers(12, 29)), int(rng.integers(12, 29))
            x0, y0 = int(rng.integers(0, W - bw + 1)), int(rng.integers(0, H - bh + 1))
            x1, y1 = x0 + bw, y0 + bh
        out[y0:y1, x0:x1] = apply_fault(out[y0:y1, x0:x1], kind, rng)
        labels.append((FAULTS.index(kind), (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H))
    return out, labels
