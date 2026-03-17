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


### ✅ AI Vision Reasoning Report (`vision_reasoning_report.py`)

Reads training logs + misclassified images, sends representative images to a
locally-hosted **Qwen2.5-VL-7B-Instruct** vision-language model, and writes a
structured Markdown reasoning report (`ai_reasoning_summary.md`).

**Do I need to start the VLM server manually?**
Yes — the script calls the VLM over HTTP (port 8000). You must start the server
in a separate terminal before running the script. See the two modes below.

---

#### Mode A — Full report (with VLM visual analysis) 🖼️

**Step 1 — Start the vLLM server** (separate terminal, keep it running):

```bash
cd ~/vision_dev_project
source venv_vision/bin/activate

VLLM_USE_V1=0 HUGGINGFACE_HUB_CACHE=”/mnt/data/pratik_models” \
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --quantization bitsandbytes \
  --gpu-memory-utilization 0.4 \
  --max-model-len 2048 \
  --enforce-eager \
  --port 8000
```

Wait until you see `Application startup complete` in the server terminal.
First run will download the model (~5.8 GB to `/mnt/data/pratik_models`).

**Step 2 — Run the report script** (your normal terminal):

```bash
cd ~/vision_dev_project/pipe
source ../venv_vision/bin/activate
python3 vision_reasoning_report.py
```

That’s it — no arguments needed. It auto-discovers the latest logs.

Output: `~/vision_dev_project/ai_reasoning_summary.md`

---

#### Mode B — Stats-only report (no GPU / no VLM server) 📊

No server needed. Skips VLM calls and produces a report with only the
numeric failure statistics and recommendations.

```bash
cd ~/vision_dev_project/pipe
source ../venv_vision/bin/activate
python3 vision_reasoning_report.py --no-vlm
```

---

#### Optional flags

```
--samples N        Images per distortion type sent to the VLM (default: 3)
--port PORT        vLLM server port (default: 8000)
--model NAME       Model name served by vLLM (default: Qwen/Qwen2.5-VL-7B-Instruct)
--logs-dir PATH    Override logs folder (default: pipe/logs)
--report PATH      Override distortion_report.json path (default: pipe/reports/distortion_report.json)
--output PATH      Override output .md path (default: ai_reasoning_summary.md in project root)
--training-log     Override: use a specific training_log JSON instead of auto-discovering latest
--misclassified    Override: use a specific misclassified JSON instead of auto-discovering latest
--no-vlm           Skip VLM calls entirely (stats-only mode)
--seed N           Random seed for image sampling (default: 42)
```

Example with all flags explicit:

```bash
python3 vision_reasoning_report.py \
  --logs-dir  logs \
  --report    reports/distortion_report.json \
  --output    ../ai_reasoning_summary.md \
  --samples   3 \
  --model     Qwen/Qwen2.5-VL-7B-Instruct \
  --port      8000
```

---

#### Prerequisites check

Before running, confirm:
1. `pipe/logs/training_log_*.json` exists (run `test.py` first)
2. `pipe/logs/misclassified_*.json` exists (run `export_misclassified.py` or check `debug_logger.py` output)
3. `pipe/reports/distortion_report.json` exists (run `distortion_diagnostic_report.py` first)
4. For Mode A: vLLM server is running and you see `Application startup complete`

---

## ▶️ Full recommended flow (all steps)

```bash
cd ~/vision_dev_project/pipe
source ../venv_vision/bin/activate

# 1. Train + log misclassifications
python3 test.py

# 2. Generate cluster diagnostic report
python3 distortion_diagnostic_report.py \
  --base-dir logs/misclassified_YYYYMMDD_HHMMSS.json \
  --output   reports/distortion_report.json \
  --plot     reports/distortion_clusters.png

# 3a. Start VLM server in a SEPARATE terminal (for full report)
#     VLLM_USE_V1=0 HUGGINGFACE_HUB_CACHE=”/mnt/data/pratik_models” \
#     python3 -m vllm.entrypoints.openai.api_server \
#       --model Qwen/Qwen2.5-VL-7B-Instruct --quantization bitsandbytes \
#       --gpu-memory-utilization 0.4 --max-model-len 2048 --enforce-eager --port 8000

# 3b. Generate AI reasoning report (once server shows “Application startup complete”)
python3 vision_reasoning_report.py

# OR stats-only without VLM:
python3 vision_reasoning_report.py --no-vlm
```

---

## 📂 Directory structure (what you should see)

```
pipe/
  best.pt                       # distortion classifier (YOLO) — Git LFS
  test.py                       # train + logging
  debug_logger.py               # Lightning callback (logging + distortion prediction)
  export_misclassified.py
  distortion_diagnostic_report.py
  vision_reasoning_report.py    # AI VLM reasoning report ← NEW

  data/                         # CIFAR-10 data (downloaded on first run)
  logs/
    training_log_*.json
    misclassified_*.json
    misclassified_*_images/     # extracted images (blur/jpeg/pixelate/noise/unknown)
  reports/
    distortion_report.json
    distortion_clusters.png

ai_reasoning_summary.md         # final report (written to project root)
```

---

## 🧪 Notes / troubleshooting

- If you run `distortion_diagnostic_report.py` with a folder like `pipe/misclassified` that has no images, it will report “No images found.” Use the `.json` log file instead.
- The script downloads ResNet18 weights on the first run (cached under `~/.cache/torch/hub`).
- The distortion predictor uses `best.pt` and is run automatically by `debug_logger.py` during training.
- `vision_reasoning_report.py` must be run from inside `pipe/` **or** from the project root — paths resolve correctly either way.
- If the vLLM server is not running and you use Mode A, you will see `Connection refused` errors per image. Use `--no-vlm` for stats-only output.
- The VLM server takes ~30–60 seconds to start. Do not run the report script until `Application startup complete` appears.
- VRAM usage at 4-bit: Qwen2.5-VL ~5.8 GB. If you hit OOM, lower `--gpu-memory-utilization` to `0.3`.
