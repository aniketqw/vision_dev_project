"""Lightning callback that records everything needed to debug a training run.

Per epoch it logs loss/accuracy/precision/recall/F1, per-class metrics, gradient norm, learning rate,
epoch time and input statistics. For every misclassified validation image it stores the image (base64 PNG),
true/predicted class and, if a tagger is given, the distortion tags. At the end it writes two files:

  training_log_<ts>.json   metrics + misclassified samples (deduplicated by image hash, sorted by count)
  misclassified_<ts>.json  the misclassified samples alone (input of the selection step)
"""
import base64
import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime
from io import BytesIO
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support

from .tagger.infer import Tagger


def _to_png_b64(img_chw_normalised: torch.Tensor):
    """[-1,1] CHW tensor -> (HxWx3 uint8 array, base64 PNG)."""
    arr = ((img_chw_normalised.numpy().transpose(1, 2, 0) + 1) / 2 * 255).round().astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return arr, base64.b64encode(buf.getvalue()).decode()


class DebugLogger(pl.Callback):
    """Args:
        save_dir: where the JSON logs are written.
        tagger:   optional `Tagger`; when given, each misclassified image is tagged.
    """

    def __init__(self, save_dir: str = "outputs/logs", tagger: Optional[Tagger] = None):
        super().__init__()
        self.save_dir, self.tagger = save_dir, tagger
        os.makedirs(save_dir, exist_ok=True)
        self.summary, self.classes = {}, {}
        self.epochs, self.batches, self.class_metrics = [], [], []
        self.grad_norms, self.lrs, self.epoch_times, self.input_stats = [], [], [], []
        self.misclassified = []
        self._preds, self._labels, self._t0 = [], [], None

    # ---- setup -----------------------------------------------------------------------------
    def on_train_start(self, trainer, pl_module):
        ds = trainer.train_dataloader.dataset
        if hasattr(ds, "classes"):
            self.classes = {i: n for i, n in enumerate(ds.classes)}
        images, _ = next(iter(trainer.train_dataloader))
        self.summary["dataset_info"] = {"classes": self.classes, "resolution": f"{images.shape[2]}x{images.shape[3]}"}

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.time()

    # ---- per validation batch --------------------------------------------------------------
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking:  # Lightning's 2-batch pre-flight check is not a real epoch
            return
        x, y = batch
        epoch = trainer.current_epoch
        self.input_stats.append({"epoch": epoch, "batch_idx": batch_idx, "input_mean": float(x.mean()),
                                 "input_std": float(x.std()), "input_min": float(x.min()), "input_max": float(x.max())})
        with torch.no_grad():
            logits = pl_module(x)
        preds = logits.argmax(1)
        self._preds += preds.cpu().tolist()
        self._labels += y.cpu().tolist()
        self.batches.append({"epoch": epoch, "batch_idx": batch_idx,
                             "batch_accuracy": float((preds == y).float().mean()),
                             "batch_loss": float(F.cross_entropy(logits, y))})

        wrong = (preds != y).nonzero().flatten().tolist()
        if wrong:
            xs = x[wrong].cpu()
            arrs, b64s = zip(*[_to_png_b64(im) for im in xs])
            tags = self.tagger.tags_batch(list(arrs)) if self.tagger else [{}] * len(arrs)
            for k, i in enumerate(wrong):
                top = max(tags[k], key=tags[k].get) if tags[k] else "none"
                self.misclassified.append({
                    "hash": hashlib.sha1(xs[k].numpy().tobytes()).hexdigest(),
                    "image_base64": b64s[k],
                    "true_label": int(y[i]), "predicted_label": int(preds[i]),
                    "distortion_predicted": top if self.tagger else "unknown",
                    "distortion_confidence": tags[k].get(top) if tags[k] else None,
                    "distortion_tags": tags[k],
                    "epoch": epoch,
                })

        self.lrs += [{"epoch": epoch, "learning_rate": float(g["lr"])} for g in trainer.optimizers[0].param_groups]
        total = sum(p.grad.norm(2).item() ** 2 for p in pl_module.parameters() if p.grad is not None) ** 0.5
        self.grad_norms.append({"epoch": epoch, "gradient_norm": float(total)})

        if batch_idx == trainer.num_val_batches[dataloader_idx] - 1:
            self._end_of_validation(trainer)

    def _end_of_validation(self, trainer):
        epoch, m = trainer.current_epoch, trainer.callback_metrics
        p, r, f1, sup = precision_recall_fscore_support(self._labels, self._preds, average=None, zero_division=0)
        self.class_metrics.append({"epoch": epoch, "precision_per_class": p.tolist(), "recall_per_class": r.tolist(),
                                   "f1_per_class": f1.tolist(), "support_per_class": sup.tolist()})
        dt = time.time() - self._t0 if self._t0 else 0.0
        self.epoch_times.append({"epoch": epoch, "time_seconds": dt})
        scalar = lambda k: float(m[k]) if k in m else 0.0
        # Lightning has not aggregated the validation metrics yet at the last batch, so compute them here
        acc = float(np.mean(np.array(self._preds) == np.array(self._labels)))
        val_loss = float(np.mean([b["batch_loss"] for b in self.batches if b["epoch"] == epoch]))
        row = {"epoch": epoch, "train_loss": scalar("train_loss"), "val_loss": val_loss,
               "accuracy": acc, "overall_accuracy": acc,
               "precision": float(p.mean()), "recall": float(r.mean()), "f1_score": float(f1.mean()),
               "num_misclassified": sum(s["epoch"] == epoch for s in self.misclassified),
               "epoch_time": dt, "learning_rate": self.lrs[-1]["learning_rate"],
               "gradient_norm": self.grad_norms[-1]["gradient_norm"]}
        self.epochs.append(row)
        print(f"Epoch {epoch}: acc={row['accuracy']:.4f} val_loss={row['val_loss']:.4f} misclassified={row['num_misclassified']}")
        self._preds, self._labels = [], []

    # ---- save ------------------------------------------------------------------------------
    def on_train_end(self, trainer, pl_module):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        counts = Counter(s["hash"] for s in self.misclassified)
        unique = {}
        for s in self.misclassified:
            unique.setdefault(s["hash"], {**s, "count": counts[s["hash"]]})
        samples = sorted(unique.values(), key=lambda s: s["count"], reverse=True)
        best = max(self.epochs, key=lambda e: e["accuracy"]) if self.epochs else {"accuracy": 0, "epoch": 0}
        log = {
            "summary": {**self.summary, "timestamp": ts, "total_epochs": len(self.epochs),
                        "best_accuracy": best["accuracy"], "best_epoch": best["epoch"],
                        "final_train_loss": self.epochs[-1]["train_loss"] if self.epochs else 0,
                        "final_val_loss": self.epochs[-1]["val_loss"] if self.epochs else 0,
                        "total_misclassified": len(self.misclassified),
                        "total_time_seconds": sum(t["time_seconds"] for t in self.epoch_times),
                        "avg_gradient_norm": float(np.mean([g["gradient_norm"] for g in self.grad_norms])) if self.grad_norms else 0},
            "epochs": self.epochs, "batch_metrics": self.batches, "class_metrics": self.class_metrics,
            "gradient_norms": self.grad_norms, "learning_rates": self.lrs, "epoch_times": self.epoch_times,
            "input_statistics": self.input_stats, "misclassified_samples": samples,
        }
        log_path = os.path.join(self.save_dir, f"training_log_{ts}.json")
        mc_path = os.path.join(self.save_dir, f"misclassified_{ts}.json")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        with open(mc_path, "w") as f:
            json.dump({"source_log": log_path, "created_at": datetime.now().isoformat(),
                       "n_misclassified": len(samples), "misclassified_samples": samples}, f)
        print(f"saved {log_path}\nsaved {mc_path}")
