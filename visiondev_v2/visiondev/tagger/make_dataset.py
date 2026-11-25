"""Build the YOLO training set from CIFAR-10 TRAIN images only (test images stay unseen for evaluation).

    python -m visiondev.tagger.make_dataset --cifar-root ./data --out outputs/tagger_dataset
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from torchvision import datasets

from ..config import TAGS
from .recipe import make_sample


def build(cifar_root: str, out: Path, n_train: int = 24000, n_val: int = 3000):
    ds = datasets.CIFAR10(cifar_root, train=True, download=True)
    order = np.random.default_rng(7).permutation(len(ds.data))
    splits = {"train": order[:n_train], "val": order[n_train:n_train + n_val]}
    stats = {}
    for split, idxs in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(1000 if split == "train" else 2000)
        counts = {"clean": 0, **{t: 0 for t in TAGS}}
        for n, i in enumerate(idxs):
            img, labels = make_sample(ds.data[i], rng)
            name = f"{split}_{n:05d}"
            cv2.imwrite(str(out / "images" / split / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in labels]
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            counts["clean"] += not labels
            for c in {l[0] for l in labels}:
                counts[TAGS[c]] += 1
        stats[split] = {"images": len(idxs), **counts}
        print(split, stats[split], flush=True)
    names = "\n".join(f"  {i}: {t}" for i, t in enumerate(TAGS))
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifar-root", default="./data")
    ap.add_argument("--out", type=Path, default=Path("outputs/tagger_dataset"))
    a = ap.parse_args()
    build(a.cifar_root, a.out)
