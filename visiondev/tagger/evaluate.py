"""Evaluate a tagger on CIFAR-10 TEST images. Writes <out>.json.

  A  held-out recipe test : injections from recipe.py, per-tag precision/recall/F1, clean specificity
  B  dose-response        : a DIFFERENT recipe (PIL, global, 5 severities) to check it generalises
  C  failed vs correct    : tag rates on images the CNN gets wrong vs right (needs --probe-json or --predictions)
  D  stored failures      : re-tag a misclassified_*.json (needs --misclassified)

    python -m visiondev.tagger.evaluate --weights weights/tagger_v2.pt --out outputs/eval_new.json \
        [--probe-json tag_probe_results.json | --predictions preds.npy] [--misclassified outputs/logs/misclassified_X.json]
"""
import argparse
import base64
import io
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from torchvision import datasets

from ..config import TAGS
from .infer import Tagger
from .recipe import make_sample

DOSE = {"blur": [0.5, 1.0, 1.5, 2.0, 2.5], "noise": [0.02, 0.05, 0.10, 0.15, 0.20],
        "jpeg": [90, 70, 50, 30, 10], "pixelate": [0.8, 0.6, 0.5, 0.4, 0.25]}


def pil_fault(kind, level, img, rng):
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=level))
    if kind == "noise":
        a = np.asarray(img, dtype=np.float32) / 255.0
        return Image.fromarray((np.clip(a + rng.normal(0, level, a.shape), 0, 1) * 255).round().astype(np.uint8))
    if kind == "jpeg":
        buf = io.BytesIO(); img.save(buf, format="JPEG", quality=int(level)); buf.seek(0)
        return Image.open(buf).convert("RGB")
    w, h = img.size
    return img.resize((max(1, int(w * level)), max(1, int(h * level))), Image.NEAREST).resize((w, h), Image.NEAREST)


def prf(truth, pred):
    res = {}
    for f in TAGS:
        tp = sum(f in t and f in p for t, p in zip(truth, pred))
        fp = sum(f not in t and f in p for t, p in zip(truth, pred))
        fn = sum(f in t and f not in p for t, p in zip(truth, pred))
        P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        res[f] = {"precision": round(P, 3), "recall": round(R, 3), "f1": round(2 * P * R / max(P + R, 1e-9), 3)}
    return res


def test_a(T, test, n=1000):
    rng = np.random.default_rng(123)
    imgs, truth = [], []
    for i in random.Random(5).sample(range(len(test.data)), n):
        im, lab = make_sample(test.data[i], rng)
        imgs.append(im); truth.append({TAGS[l[0]] for l in lab})
    pred = [set(d) for d in T.tags_batch(imgs)]
    clean = [k for k, t in enumerate(truth) if not t]
    single = [k for k, t in enumerate(truth) if len(t) == 1]
    mixed = [k for k, t in enumerate(truth) if len(t) >= 2]
    return {"n": n, "per_fault": prf(truth, pred),
            "clean_specificity(no detection)": round(float(np.mean([not pred[k] for k in clean])), 3),
            "exact_set_match": {"single": round(float(np.mean([pred[k] == truth[k] for k in single])), 3),
                                "mixed": round(float(np.mean([pred[k] == truth[k] for k in mixed])), 3)},
            "n_clean_single_mixed": [len(clean), len(single), len(mixed)]}


def test_b(T, test, n=200, levels=(1, 2, 3, 4, 5)):
    idx = random.Random(0).sample(range(len(test.data)), n)
    out = {}
    for kind, vals in DOSE.items():
        rng = np.random.default_rng(1)
        for L in levels:
            arrs = [np.asarray(pil_fault(kind, vals[L - 1], Image.fromarray(test.data[i]), rng)) for i in idx]
            dets = T.tags_batch(arrs)
            tags = Counter(t for d in dets for t in d)
            out[f"{kind}@{L}"] = {"recall": round(tags[kind] / len(dets), 3),
                                  "none": round(sum(not d for d in dets) / len(dets), 3),
                                  "only_correct_tag": round(sum(set(d) == {kind} for d in dets) / len(dets), 3),
                                  "tag_counts": dict(tags)}
    return out


def failed_vs_correct_indices(test, predictions_npy, n=1500, seed=0):
    """predictions_npy: .npy of your model's predicted labels for the 10,000 CIFAR-10 test images."""
    pred = np.load(predictions_npy)
    wrong = np.where(pred != np.array(test.targets))[0].tolist()
    right = np.where(pred == np.array(test.targets))[0].tolist()
    r = random.Random(seed)
    return {"fail": r.sample(wrong, min(n, len(wrong))), "correct": r.sample(right, min(n, len(right)))}


def test_c(T, test, groups):
    from scipy.stats import chi2_contingency
    sets = {g: [set(d) for d in T.tags_batch([test.data[i] for i in ix])] for g, ix in groups.items()}
    n = len(sets["fail"]); out = {"n_per_group": n, "tags": {}}
    for name in TAGS + ["none"]:
        f = sum((name in s) if name != "none" else (not s) for s in sets["fail"])
        c = sum((name in s) if name != "none" else (not s) for s in sets["correct"])
        p = float(chi2_contingency([[f, n - f], [c, n - c]])[1]) if 0 < f + c < 2 * n else 1.0
        out["tags"][name] = {"failed_rate": round(f / n, 3), "correct_rate": round(c / n, 3),
                             "lift": round((f / n) / (c / n), 3) if c else None, "p_value": round(p, 4)}
    return out


def test_d(T, mc_path):
    mc = json.loads(Path(mc_path).read_text())
    mc = mc["misclassified_samples"] if isinstance(mc, dict) else mc
    arrs = [np.asarray(Image.open(io.BytesIO(base64.b64decode(m["image_base64"]))).convert("RGB")) for m in mc]
    dets = T.tags_batch(arrs)
    old = [m.get("distortion_predicted") for m in mc]
    top = [max(d, key=d.get) if d else "none" for d in dets]
    return {"n_unique": len(mc), "new_top_tag_unique": dict(Counter(top)), "old_tag_unique": dict(Counter(old)),
            "any_fault_flagged": round(float(np.mean([bool(d) for d in dets])), 3),
            "crosstab_old_to_new": {o: dict(Counter(t for oo, t in zip(old, top) if oo == o)) for o in sorted(set(old), key=str)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifar-root", default="./data")
    ap.add_argument("--imgsz", type=int, default=64)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--probe-json", type=Path, help="JSON with rows.fail/rows.correct lists of {'idx': ...}")
    ap.add_argument("--predictions", help=".npy of your model's predicted labels for the CIFAR-10 test set (for test C)")
    ap.add_argument("--misclassified", type=Path)
    a = ap.parse_args()
    T = Tagger(a.weights, imgsz=a.imgsz, conf=a.conf)
    test = datasets.CIFAR10(a.cifar_root, train=False, download=True)
    res = {"weights": str(a.weights), "conf": a.conf, "imgsz": a.imgsz,
           "A_heldout_recipe": test_a(T, test), "B_dose_response": test_b(T, test)}
    groups = None
    if a.probe_json:
        rows = json.loads(a.probe_json.read_text())["rows"]
        groups = {g: [r["idx"] for r in rs] for g, rs in rows.items()}
    elif a.predictions:
        groups = failed_vs_correct_indices(test, a.predictions)
    if groups:
        res["C_failed_vs_correct"] = test_c(T, test, groups)
    if a.misclassified:
        res["D_stored_failures"] = test_d(T, a.misclassified)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("saved", a.out)


if __name__ == "__main__":
    main()
