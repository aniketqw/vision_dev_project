# VisionDev

Find out why an image classifier fails. Add one callback to your PyTorch Lightning training run. It tags every
misclassified image with a distortion tagger, groups the failures by tag, picks representative images, asks a
vision-language model about them and writes a diagnostic report.

```python
import pytorch_lightning as pl
from visiondev import DebugLogger

trainer = pl.Trainer(max_epochs=3, callbacks=[DebugLogger(diagnose=True)])
trainer.fit(model, train_loader, val_loader)      # your model, your data
# -> visiondev_runs/<timestamp>/report.md
```

## What happens

```
training (your model)
  └─ DebugLogger callback, every validation batch
       ├─ logs metrics, per-class scores, gradient norm, learning rate
       └─ tags each misclassified image: blur | noise | pixelate | jpeg | none     (visiondev/tagger)
training ends
  ├─ saves training_log_<ts>.json and misclassified_<ts>.json
  ├─ stores the failure images grouped by tag:  misclassified_<ts>_images/<tag>/<hash>.png
  └─ diagnose=True:
       ├─ ResNet18 features → PCA(50) → t-SNE, per tag the 3 most typical + 3 outlier images   (visiondev/selection)
       ├─ those images + failure statistics → vision-language model, with a verification step  (visiondev/diagnosis)
       └─ report.md   ← the deliverable
```

The run folder (`visiondev_runs/<timestamp>/`) holds everything: `report.md`, the grouped images, `selection/tsne.png`
(cluster map), `selection/distortion_report.json` (the picked images) and the JSON logs.

## Install

```bash
pip install -e .                  # callback + tagger (weights are bundled in the package)
pip install -e ".[diagnosis]"     # adds the VLM report step
```

`DebugLogger()` without `diagnose=True` needs only the base install: it still tags and groups the images.

## Use

Your batches must be `(images, labels)` with RGB images normalised to mean 0.5 / std 0.5 (the callback
un-normalises them to save PNGs). A full runnable example with a small CIFAR-10 CNN is in
[`examples/cifar10_simplecnn.py`](examples/cifar10_simplecnn.py).

```python
from visiondev import DebugLogger
from visiondev.pipeline import DiagnosisConfig

cb = DebugLogger(
    save_dir="runs/exp1",                 # default: visiondev_runs/<timestamp>
    tagger="default",                     # bundled weights | path to .pt | None to skip tagging
    diagnose=True,
    diagnosis=DiagnosisConfig(vlm_port=8000, vlm_model="Qwen/Qwen2.5-VL-7B-Instruct", samples=6),
)
```

The VLM is any OpenAI-compatible endpoint at `http://localhost:<vlm_port>/v1` (for example vLLM). With Ollama, run
`visiondev-ollama-proxy --upstream http://localhost:11434 --model qwen2.5vl:7b` first. If no endpoint answers, the
report is still written with statistics only. If the report step fails, the training logs are already saved and
`visiondev diagnose <run_dir>` re-runs it:

```bash
visiondev diagnose visiondev_runs/20260101_120000 --port 8000 --model Qwen/Qwen2.5-VL-7B-Instruct
visiondev diagnose <run_dir> --no-vlm        # statistics only
```

## The tagger: read this before trusting the tags

The tagger is a YOLO11n detector, retrained for 32x32 images. It is accurate on injected faults (91-95% recall per
distortion on held-out images, 97.6% of clean images left untagged) and generalises to a second injection recipe.
Details, tests and how to retrain: [`results/tagger/README.md`](results/tagger/README.md) and
[`notebooks/tagger_v2.ipynb`](notebooks/tagger_v2.ipynb).

On **natural** CIFAR-10 failures it answers "none" for about 96% of images, and tagged failures are no more common than
tagged correct images. So on clean data the four distortions mostly do not explain failures; those are mainly
semantic confusion (cat/dog, truck/car), label noise and model capacity. `none` is therefore a bucket of its own: it is
clustered, sent to the VLM with prompts about ambiguity, pose, crop and likely label problems, and gets its own
recommendation (audit labels and confusions, not augmentation).

## Layout

```
visiondev/
  callback.py        DebugLogger: the entry point
  pipeline.py        diagnose(): selection + VLM + report
  cli.py             `visiondev diagnose`
  config.py  log.py
  tagger/            infer.py (Tagger) · recipe.py, make_dataset.py, train.py, evaluate.py (retrain and evaluate)
  selection/         archetypes.py: group by tag, ResNet18 → PCA → t-SNE, typical + outlier
  diagnosis/         VLM agent (observe → hypothesise → verify → conclude), tools, schemas, renderer, optional RAG memory
  weights/           tagger_v2.pt (Git LFS)
  ollama_proxy.py    OpenAI-style endpoint in front of Ollama
examples/            cifar10_simplecnn.py
notebooks/           tagger_v2.ipynb (executed, outputs stored)
results/tagger/      tagger results, training curves, logs
rag_kb/              15 fault → fix cards for the planned fix-recommendation step (not wired in yet)
```

`visiondev/weights/tagger_v2.pt` is stored with Git LFS; install `git-lfs` before cloning, or the file will be a small pointer.
