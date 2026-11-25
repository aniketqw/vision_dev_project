"""Pick representative failures per distortion tag.

    misclassified_*.json
      -> save images into <name>_images/<tag>/<hash>.png
      -> ResNet18 features (ImageNet weights, classifier removed)
      -> PCA to 50 dims -> t-SNE to 2 dims
      -> per tag: the 3 images closest to the tag's centroid ("typical")
                  and the 3 furthest from it ("outlier")
      -> reports/distortion_report.json + a cluster plot

The JSON report is what the diagnosis step reads to decide which images to show the VLM.
"""
import base64
import datetime
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ..config import ANALYSIS_TYPES
from ..log import logger

CATEGORIES = ANALYSIS_TYPES  # four distortions + "none" (tagger found no fault)


def extract_images(mc_json: Path, categories=CATEGORIES) -> Path:
    """Write each misclassified image to <mc_json stem>_images/<tag>/<hash>.png and return that folder."""
    out = mc_json.parent / f"{mc_json.stem}_images"
    samples = json.loads(mc_json.read_text()).get("misclassified_samples", [])
    if not samples:
        raise SystemExit(f"No misclassified samples in {mc_json}")
    for c in categories:
        (out / c).mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(samples):
        tag = s.get("distortion_predicted") or "unknown"
        if tag not in categories or not s.get("image_base64"):
            continue
        p = out / tag / f"{s.get('hash') or f'img_{i:06d}'}.png"
        if not p.exists():
            p.write_bytes(base64.b64decode(s["image_base64"]))
    return out


def collect_paths(images_dir: Path, categories, max_per_category: int):
    paths, labels = [], []
    for c in categories:
        files = sorted((images_dir / c).glob("*.png"))[:max_per_category]
        paths += files
        labels += [c] * len(files)
    return paths, labels


def resnet_features(paths: List[Path], device, batch_size: int = 64) -> np.ndarray:
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net = nn.Sequential(*list(net.children())[:-1]).eval().to(device)
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]]).to(device)
            feats.append(net(batch).flatten(1).cpu().numpy())
    return np.concatenate(feats)


def embed_2d(features: np.ndarray, seed: int = 42) -> np.ndarray:
    n_pca = min(50, features.shape[0], features.shape[1])
    reduced = PCA(n_components=n_pca).fit_transform(features)
    perplexity = min(30, max(2, (len(reduced) - 1) // 3))
    return TSNE(n_components=2, init="pca", random_state=seed, learning_rate="auto",
                perplexity=perplexity).fit_transform(reduced)


def pick_archetypes(emb: np.ndarray, labels: List[str], paths: List[Path], categories, k: int = 3) -> Dict:
    out = {}
    for c in categories:
        idx = [i for i, l in enumerate(labels) if l == c]
        if not idx:
            out[c] = {"typical": [], "outlier": []}
            continue
        d = np.linalg.norm(emb[idx] - emb[idx].mean(0), axis=1)
        order = np.argsort(d)
        kk = min(k, len(order))
        out[c] = {"typical": [{"file": str(paths[idx[i]]), "distance": float(d[i])} for i in order[:kk]],
                  "outlier": [{"file": str(paths[idx[i]]), "distance": float(d[i])} for i in order[-kk:][::-1]],
                  "note": "typical = closest to the tag's centroid; outlier = furthest (mixed, ambiguous or odd cases)"}
    return out


def plot(emb, labels, paths, archetypes, out_png: Path):
    cats = sorted(set(labels)); cmap = plt.get_cmap("tab10")
    col = {c: cmap(i % 10) for i, c in enumerate(cats)}
    pos = {str(p): i for i, p in enumerate(paths)}
    plt.figure(figsize=(11, 9), dpi=150)
    for c in cats:
        m = [i for i, l in enumerate(labels) if l == c]
        plt.scatter(emb[m, 0], emb[m, 1], s=16, alpha=0.6, color=col[c], label=c, edgecolors="none")
    for c, ar in archetypes.items():
        for kind, marker in (("typical", "*"), ("outlier", "X")):
            for e in ar[kind]:
                i = pos.get(e["file"])
                if i is not None:
                    plt.scatter(emb[i, 0], emb[i, 1], marker=marker, s=170, color=col[c], edgecolor="k", linewidth=0.8)
    plt.title("Misclassified images by tag (t-SNE of ResNet18 features); star = typical, X = outlier")
    plt.legend(title="tag", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True); plt.savefig(out_png); plt.close()


def run(mc_json: Path, report_path: Path, plot_path: Path, max_per_category: int = 300, k: int = 3,
        categories=CATEGORIES) -> Dict:
    images_dir = extract_images(mc_json, categories)
    paths, labels = collect_paths(images_dir, categories, max_per_category)
    if len(paths) < 5:
        raise SystemExit("Too few tagged images to cluster.")
    logger.info(f"{len(paths)} images: " + ", ".join(f"{c}={labels.count(c)}" for c in categories))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = embed_2d(resnet_features(paths, device))
    archetypes = pick_archetypes(emb, labels, paths, categories, k)
    plot(emb, labels, paths, archetypes, plot_path)
    report = {"archetypes": archetypes, "generated_at": datetime.datetime.now().isoformat(),
              "counts": {c: labels.count(c) for c in categories}}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report
