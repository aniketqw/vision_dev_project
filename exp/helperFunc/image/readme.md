# 🚀 vLLM + Qwen2.5-VL Setup Guide (GPU + Large Disk)

This document describes a stable and disk-safe setup for running **vLLM** with **Qwen/Qwen2.5-VL-7B-Instruct** on an RTX 3090 (24GB VRAM) while ensuring all large downloads go to a dedicated 1.9TB disk.

---

## 🔹 Step 0: Activate Virtual Environment

```bash
source venv_vision/bin/activate
```

---

## 🔹 Step 1: Start vLLM Server (Basic)

⚠️ **First run may take several minutes to download the model.**

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --trust-remote-code \
  --gpu-memory-utilization 0.5 \
  --max-model-len 4096 \
  --port 8000
```

**Adjust process priority (optional):**
```bash
sudo renice -n 15 -p 271230
```

---

## 🔹 Alternative: Run with `nice` + Quantization

```bash
sudo nice -n -10 python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --quantization bitsandbytes \
  --gpu-memory-utilization 0.4 \
  --port 8000
```

---

## 🔹 Step 2: Create Dedicated Model Storage (Large Disk)

```bash
mkdir -p /mnt/data/pratik_models
sudo chown -R pratik2:pratik2 /mnt/data/pratik_models
```

This ensures Hugging Face models are downloaded to the 1.9TB disk instead of your home partition.

---

## 🔹 Step 3: Permanent Fix — Move Project to Large Drive

Since the home directory is on a small partition, move the project and create a symbolic link.

**1️⃣ Move the project:**
```bash
sudo mv /home/pratik2/vision_dev_project /mnt/data/
```

**2️⃣ Fix ownership:**
```bash
sudo chown -R pratik2:pratik2 /mnt/data/vision_dev_project
```

**3️⃣ Create symlink:**
```bash
ln -s /mnt/data/vision_dev_project /home/pratik2/vision_dev_project
```

The system now behaves as if the project is in `$HOME`, but it is physically stored on the large disk.

---

## 🔹 Step 4: Launch vLLM Using the Big Drive

⚠️ **`sudo` resets environment variables. Pass the cache path explicitly.**

**Recommended (Stable):**
```bash
sudo nice -n -10 env HUGGINGFACE_HUB_CACHE="/mnt/data/pratik_models" \
/home/pratik2/vision_dev_project/venv_vision/bin/python3 \
-m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen2.5-VL-7B-Instruct \
--quantization bitsandbytes \
--gpu-memory-utilization 0.3 \
--max-model-len 4096 \
--port 8000
```

---

## 🔹 Lower VRAM Usage (Extreme Safety Mode)

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --quantization bitsandbytes \
  --gpu-memory-utilization 0.2 \
  --max-model-len 1024 \
  --max-num-seqs 1 \
  --port 8000
```

---

## 🔹 Most Stable Configuration (Disable CUDA Graphs)

Uses vLLM V0 engine and eager execution to save ~2–3GB VRAM.

```bash
VLLM_USE_V1=0 env HUGGINGFACE_HUB_CACHE="/mnt/data/pratik_models" \
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --quantization bitsandbytes \
  --gpu-memory-utilization 0.4 \
  --max-model-len 2048 \
  --enforce-eager \
  --port 8000
```

---

## 🔹 Git Configuration

**Set repository identity:**
```bash
git config --local user.name "aniketqw"
git config --local user.email "38223792+aniketqw@users.noreply.github.com"
```

**Cache credentials for 1 hour:**
```bash
git config --local credential.helper 'cache --timeout=3600'
```

---

## 🔹 Fix: "Dubious Ownership" Git Error

This occurs after moving the repository with `sudo`.

**✅ Register the directory as safe:**
```bash
git config --global --add safe.directory /mnt/data/vision_dev_project
```

**🔍 Why this happens:**
- Project was moved using `sudo`
- Ownership / filesystem metadata changed
- Git blocks access as a security precaution
- Explicitly marking the directory as safe resolves it

---

## ✅ Final Notes
- ✔ Models stored on 1.9TB disk
- ✔ VRAM usage controlled for RTX 3090
- ✔ Stable vLLM launch configs included
- ✔ Git security issue resolved
- ✔ Reproducible and clean setup

---
python3 -m exp.helperFunc.image.image_test