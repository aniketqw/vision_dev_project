# 📌 `pipe/` — What this folder does

This folder contains the training/analysis pipeline used to:

- Train a CIFAR-10 model and log misclassified samples
- Predict the type of distortion on each misclassified image (blur/jpeg/pixelate/noise)
- Export misclassified images and run clustering diagnostics

---

## 🧠 What’s implemented (current capabilities)

### ✅ Training + misclassification logging
- `test.py` runs a PyTorch Lightning training loop on CIFAR-10.
- `pipe/debug_logger.py` is a Lightning callback that:
  - records per-epoch stats (loss/accuracy/time)
  - collects misclassified samples (image base64 + true/pred labels)
  - runs a distortion classifier (Ultralytics YOLO `best.pt`) and stores:
    - `distortion_predicted` (blur/jpeg/pixelate/noise)
    - `distortion_confidence`

Output:
- `logs/training_log_*.json` (full training metrics + misclassified samples)
- `logs/misclassified_*.json` (exported misclassified samples only)

---

### ✅ Misclassified export
- `export_misclassified.py` reads a training log JSON and writes a smaller JSON with just the misclassified samples.

Example:
```bash
cd pipe
python export_misclassified.py \
  --input logs/training_log_20260317_005331.json \
  --output logs/misclassified_20260317_005331.json
```

---

### ✅ Diagnostic report (clustering + archetypes)
- `distortion_diagnostic_report.py` creates a report from misclassified images.
- It can **take either**:
  1) a directory with per-distortion subfolders (blur/jpeg/pixelate/noise)
  2) a misclassified JSON log (`logs/misclassified_*.json`) — it auto-extracts the images

It produces:
- `reports/distortion_report.json` — archetypes (typical/outlier images + distances)
- `reports/distortion_clusters.png` — t-SNE scatterplot (colored by distortion)

---

## ▶️ How to run (recommended flow)

### 1) Train + log misclassifications

```bash
cd pipe
source ../venv_vision/bin/activate
python test.py
```

After this finishes you should have:
- `logs/training_log_*.json`
- `logs/misclassified_*.json`

### 2) Generate diagnostic report

```bash
cd pipe
source ../venv_vision/bin/activate
python distortion_diagnostic_report.py \
  --base-dir logs/misclassified_YYYYMMDD_HHMMSS.json \
  --output reports/distortion_report.json \
  --plot reports/distortion_clusters.png
```

✅ This will automatically extract the images from the JSON log and run the clustering.

---

## 📂 Directory structure (what you should see)

```
pipe/
  best.pt               # distortion classifier (YOLO) tracked by Git LFS
  test.py               # train + logging
  debug_logger.py       # Lightning callback for logging + distortion prediction
  export_misclassified.py
  distortion_diagnostic_report.py

  data/                 # CIFAR-10 data (downloaded by training)
  logs/
    training_log_*.json
    misclassified_*.json
    misclassified_*_images/  # extracted images (blur/jpeg/pixelate/noise/unknown)
  reports/
    distortion_report.json
    distortion_clusters.png
```

---

## 🧪 Notes / troubleshooting

- If you run `distortion_diagnostic_report.py` with a folder like `pipe/misclassified` that has no images, it will report “No images found.” Use the `.json` log file instead.
- The script downloads ResNet18 weights on the first run (cached under `~/.cache/torch/hub`).
- The distortion predictor uses `best.pt` and is run automatically by `debug_logger.py` during training.

---

## 🧩 What’s next (optional enhancements)

If you want, I can add:
- 🗂 Automatic “latest log” detection (auto-select newest `training_log_*.json`)
- 📦 Export the 24 archetype images into a dedicated `reports/archetypes/` folder
- 🎯 Filter by `distortion_confidence` (e.g., ignore low-confidence predictions)
- 🧊 Support for larger datasets or custom distortion categories
