"""Generate a diagnostic report for distortion misclassifications.

This script expects a directory structure like:

  /path/to/misclassified/
      blur/
      jpeg/
      pixelate/
      noise/

Each folder should contain image files (PNG/JPEG/etc.) representing misclassified
images annotated with that distortion.

It produces:
  - a scatter plot of all images embedded in 2D (PCA -> t-SNE)
  - selection of 3 "typical" (closest-to-centroid) and 3 "outlier" (furthest-from-centroid)
    images per distortion category
  - a JSON report describing those 24 images and their distances

Usage:
  python distortion_diagnostic_report.py \
    --base-dir /path/to/misclassified \
    --output report.json \
    --plot output_plot.png

Dry-run mode:
  python distortion_diagnostic_report.py --dry-run \
    --base-dir /path/to/misclassified

"""

import argparse
import base64
import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _get_image_paths(base_dir: Path, categories: List[str]) -> Dict[str, List[Path]]:
    out = {}
    for cat in categories:
        p = base_dir / cat
        if not p.exists() or not p.is_dir():
            out[cat] = []
            continue
        out[cat] = sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    return out


def _load_and_preprocess(img_path: Path, transform) -> torch.Tensor:
    img = Image.open(img_path).convert('RGB')
    return transform(img)


def _extract_images_from_log(
    json_path: Path,
    out_dir: Path,
    categories: List[str],
) -> Path:
    """Extract base64-encoded images from a misclassified JSON log.

    Images are placed into subfolders under `out_dir` named by their predicted
    distortion label (fallbacking to 'unknown').
    """
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    samples = data.get("misclassified_samples", [])
    if not samples:
        raise SystemExit(f"No misclassified samples found in: {json_path}")

    # ensure subfolders exist
    for cat in categories + ["unknown"]:
        (out_dir / cat).mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(samples):
        b64 = sample.get("image_base64")
        if not b64:
            continue

        distortion = sample.get("distortion_predicted") or sample.get("distortion_type") or "unknown"
        if distortion not in categories:
            distortion = "unknown"

        fname = sample.get("hash") or f"img_{idx:06d}"
        out_path = out_dir / distortion / f"{fname}.png"
        if out_path.exists():
            continue

        try:
            img_bytes = base64.b64decode(b64)
            with open(out_path, "wb") as f:
                f.write(img_bytes)
        except Exception:
            # skip problematic samples
            continue

    return out_dir


def _extract_features(
    image_paths: List[Path],
    model: nn.Module,
    transform,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    model.eval()
    features = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_imgs = torch.stack([_load_and_preprocess(p, transform) for p in batch_paths]).to(device)
            out = model(batch_imgs)
            features.append(out.cpu().numpy())
    return np.concatenate(features, axis=0) if features else np.zeros((0, model.fc.in_features))


def _select_archetypes(
    embeddings: np.ndarray,
    labels: List[str],
    categories: List[str],
    paths: List[Path],
    choose_k: int = 3,
) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
    report = {}
    for cat in categories:
        idx = [i for i, l in enumerate(labels) if l == cat]
        if len(idx) == 0:
            report[cat] = {
                'typical': [],
                'outlier': [],
            }
            continue

        cat_embeddings = embeddings[idx]
        centroid = np.mean(cat_embeddings, axis=0)
        dists = np.linalg.norm(cat_embeddings - centroid[None, :], axis=1)
        order = np.argsort(dists)

        # handle limited case
        n = len(order)
        k = min(choose_k, n)

        typical_idx = order[:k]
        outlier_idx = order[-k:][::-1]

        report[cat] = {
            'typical': [
                (str(paths[idx[i]]), float(dists[i])) for i in typical_idx
            ],
            'outlier': [
                (str(paths[idx[i]]), float(dists[i])) for i in outlier_idx
            ],
        }
    return report


def _draw_cluster_plot(
    embeddings: np.ndarray,
    labels: List[str],
    archetypes: Dict[str, Dict[str, List[Tuple[str, float]]]],
    all_paths_map: Dict[Path, int],
    output_path: Path,
    dpi: int = 200,
):
    plt.figure(figsize=(12, 10), dpi=dpi)

    unique_labels = sorted(set(labels))
    palette = plt.get_cmap('tab10')

    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    xs = embeddings[:, 0]
    ys = embeddings[:, 1]
    cs = [palette(label_to_idx[l] % 10) for l in labels]

    plt.scatter(xs, ys, c=cs, s=18, alpha=0.7, edgecolors='none')

    # overlay archetypes
    for cat, ar in archetypes.items():
        color = palette(label_to_idx.get(cat, 0) % 10)
        for typ in ar.get('typical', []):
            # find index in the global list
            fp = Path(typ[0])
            if fp in all_paths_map:
                i = all_paths_map[fp]
                plt.scatter(xs[i], ys[i], marker='*', s=150, c=[color], edgecolor='k', linewidth=0.7)
        for out in ar.get('outlier', []):
            fp = Path(out[0])
            if fp in all_paths_map:
                i = all_paths_map[fp]
                plt.scatter(xs[i], ys[i], marker='X', s=150, c=[color], edgecolor='k', linewidth=0.7)

    plt.xlabel('PCA component 1')
    plt.ylabel('PCA component 2')
    plt.title('Distortion cluster map (t-SNE on ResNet features)')
    plt.legend(unique_labels, title='Distortion', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _build_report(
    archetypes: Dict[str, Dict[str, List[Tuple[str, float]]]],
) -> Dict:
    summary = {}
    for cat, ar in archetypes.items():
        summary[cat] = {
            'typical': [
                {'file': p, 'distance': d} for p, d in ar.get('typical', [])
            ],
            'outlier': [
                {'file': p, 'distance': d} for p, d in ar.get('outlier', [])
            ],
            'note': (
                "Typical images represent the core of the distortion cluster (closest to centroid). "
                "Outliers are those that deviate the most; they are either mislabeled or show mixed/ambiguous distortions."
            ),
        }
    return {'archetypes': summary, 'generated_at': datetime.datetime.now().isoformat()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a distortion diagnostic report.')
    parser.add_argument(
        '--base-dir',
        required=True,
        type=Path,
        help='Base directory containing distortion subfolders (blur/jpeg/pixelate/noise), or a misclassified JSON log file',
    )
    parser.add_argument('--output', required=True, type=Path, help='Output JSON report file')
    parser.add_argument('--plot', required=True, type=Path, help='Output cluster plot image (PNG)')
    parser.add_argument('--dry-run', action='store_true', help='Run without requiring many images (uses what is available)')
    parser.add_argument('--max-samples', type=int, default=1000, help='Max total images to process (for speed)')
    args = parser.parse_args()

    # Ensure output directories exist
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.plot.parent.mkdir(parents=True, exist_ok=True)

    categories = ['blur', 'jpeg', 'pixelate', 'noise']

    # If a JSON log is provided, extract images into a temporary structure first.
    if args.base_dir.is_file() and args.base_dir.suffix.lower() == '.json':
        extracted_dir = args.base_dir.parent / f"{args.base_dir.stem}_images"
        print(f"Extracting images from misclassified log: {args.base_dir} -> {extracted_dir}")
        args.base_dir = _extract_images_from_log(args.base_dir, extracted_dir, categories)

    images = _get_image_paths(args.base_dir, categories)

    # Build global list, controlling max samples
    global all_paths_map
    all_paths = []
    all_labels = []
    for cat in categories:
        cat_paths = images.get(cat, [])
        if args.max_samples and len(all_paths) < args.max_samples:
            remaining = args.max_samples - len(all_paths)
            cat_paths = cat_paths[:remaining]
        all_paths.extend(cat_paths)
        all_labels.extend([cat] * len(cat_paths))

    if not all_paths:
        raise SystemExit('No images found. Ensure the directory structure and image files exist.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Pretrained feature extractor (ResNet18 without final classification)
    model = models.resnet18(pretrained=True)
    model = nn.Sequential(*(list(model.children())[:-1]))  # remove classifier
    model.eval().to(device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Extract features
    features = _extract_features(all_paths, model, transform, device)
    features = features.reshape(features.shape[0], -1)

    # Reduce dimensions: PCA -> t-SNE
    pca = PCA(n_components=50)
    pca_feats = pca.fit_transform(features)

    tsne = TSNE(n_components=2, init='pca', random_state=42, learning_rate='auto')
    embeddings = tsne.fit_transform(pca_feats)

    # Build index map for plotting
    all_paths_map = {p: i for i, p in enumerate(all_paths)}

    archetypes = _select_archetypes(embeddings, all_labels, categories, all_paths)

    _draw_cluster_plot(embeddings, all_labels, archetypes, all_paths_map, args.plot)

    report = _build_report(archetypes)

    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)

    print('Report written to:', args.output)
    print('Cluster plot written to:', args.plot)

# python pipe/distortion_diagnostic_report.py \
#   --base-dir pipe/misclassified \
#   --output pipe/reports/distortion_report.json \
#   --plot pipe/reports/distortion_clusters.png