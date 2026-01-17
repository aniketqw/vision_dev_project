
# Training Analytics Package

Comprehensive logging and debugging callback for PyTorch Lightning training with AI-powered failure analysis.

## 📁 File Structure

```
training_analytics/
├── __init__.py
├── debug_logger.py      # PyTorch Lightning callback for metrics logging
└── generate_report.py   # AI Reasoning report generator
```

## 🚀 Usage

### 1. Training with DebugLogger

Add the `DebugLogger` to your Lightning trainer. It will automatically save training metrics and misclassified samples (as Base64) to your logs directory.

```python
from exp.helperFunc.training_analytics.debug_logger import DebugLogger
import pytorch_lightning as pl

# Metrics will be saved to your specified path
debug_logger = DebugLogger(save_dir='/mnt/data/vision_dev_project/logs')

trainer = pl.Trainer(max_epochs=10, callbacks=[debug_logger])
trainer.fit(model, train_loader, val_loader)
```

### 🧠 2. Start Reasoning Server (vLLM)

Before running the report generator, launch the local reasoning server in a **separate terminal**. This server hosts the Molmo-7B model for AI-powered analysis.

```bash
# Start vLLM server for Molmo-7B reasoning
export VLLM_USE_V1=0
export HUGGINGFACE_HUB_CACHE="/mnt/data/pratik_models"

python3 -m vllm.entrypoints.openai.api_server \
  --model allenai/Molmo-7B-D-0924 \
  --trust-remote-code \
  --quantization bitsandbytes \
  --gpu-memory-utilization 0.35 \
  --max-model-len 2048 \
  --enforce-eager \
  --port 8000
```

**Server Verification:**
```bash
# Check if server is running
curl http://localhost:8000/v1/models
```

### 📊 3. Generate AI-Powered Analysis Report

Run the report generator from your project root. It will:
1. Find the latest training log in `exp/helperFunc/logs/`
2. Send misclassified samples to the Molmo-7B model via vLLM API
3. Generate a comprehensive failure analysis report

```bash
# From vision_dev_project root directory:
python3 -m exp.helperFunc.training_analytics.generate_report
```

**Expected Output:**
```
📄 Analyzing latest log: exp/helperFunc/logs/training_log_20250117_230000.json
✅ Report saved to /mnt/data/vision_dev_reports/ai_analysis_230500.txt
```

## 📋 Report Content Example

The generated report contains AI analysis of model failures:

```
=== AI FAILURE ANALYSIS REPORT ===
Generated: 2025-01-17 23:05:00
Source Log: training_log_20250117_230000.json
Total Samples Analyzed: 5

SAMPLE 1
────────
Ground Truth: "cat"
Predicted: "dog"
Confidence: 0.67
Epoch: 5 | Batch: 32

AI ANALYSIS:
The model likely confused the feline subject with a dog due to similar fur texture
patterns under this lighting condition. The ear shape is ambiguous because of
the 45-degree angle. Consider adding more varied angles and lighting conditions
to your training data augmentation pipeline.

SAMPLE 2
────────
Ground Truth: "stop sign"
Predicted: "yield sign"
Confidence: 0.72
Epoch: 7 | Batch: 15

AI ANALYSIS:
Partial occlusion by tree branches has reduced the distinctive octagonal shape
recognition. The red color and text are visible but the shape context is lost.
Recommend adding occlusion augmentation techniques to improve robustness.
```

## 🔧 Configuration

### Log Directory Structure
By default, the system expects:
- **Input Logs:** `exp/helperFunc/logs/` (auto-created by DebugLogger)
- **Output Reports:** `/mnt/data/vision_dev_reports/` (auto-created)

### Customizing Paths
Modify these variables in `generate_report.py`:
```python
LOG_DIR = "exp/helperFunc/logs"  # Change to your log directory
REPORT_OUTPUT_DIR = "/mnt/data/vision_dev_reports"  # Change output location
VLLM_URL = "http://localhost:8000/v1/chat/completions"  # Change if different port
```

## 📊 JSON Log Structure

The DebugLogger creates structured JSON files with:
```json
{
  "summary": {
    "timestamp": "20260117_230000",
    "total_epochs": 10,
    "best_accuracy": 0.85,
    "best_epoch": 7,
    "final_train_loss": 0.234,
    "final_val_loss": 0.456,
    "total_misclassified": 1500
  },
  "epochs": [...],
  "misclassified_samples": [
    {
      "epoch": 5,
      "batch_idx": 32,
      "true_label": "cat",
      "predicted_label": "dog",
      "confidence": 0.67,
      "image": "base64_encoded_string_here"
    }
  ]
}
```

## ⚡ Quick Start Script

Create a `run_analysis.sh` script for one-command execution:

```bash
#!/bin/bash
# run_analysis.sh

echo "🚀 Starting complete training analysis pipeline..."

# Step 1: Train model with analytics
echo "📊 Training model with DebugLogger..."
python3 train_model.py

# Step 2: Start vLLM server in background
echo "🧠 Starting Molmo-7B reasoning server..."
python3 -m vllm.entrypoints.openai.api_server \
  --model allenai/Molmo-7B-D-0924 \
  --quantization bitsandbytes \
  --gpu-memory-utilization 0.35 \
  --max-model-len 2048 \
  --port 8000 &
SERVER_PID=$!

# Wait for server to start
sleep 10

# Step 3: Generate report
echo "📊 Generating AI analysis report..."
python3 -m exp.helperFunc.training_analytics.generate_report

# Cleanup
echo "🧹 Stopping server..."
kill $SERVER_PID

echo "✅ Analysis complete!"
```

## 🐛 Troubleshooting

### Common Issues & Solutions

**1. vLLM Server Won't Start:**
```bash
# Check GPU memory
nvidia-smi

# Try lower memory utilization
--gpu-memory-utilization 0.25  # Reduce from 0.35
```

**2. No Log Files Found:**
- Ensure DebugLogger is added to trainer callbacks
- Check `save_dir` path exists and is writable
- Verify training completed successfully

**3. AI Analysis Fails:**
```bash
# Test vLLM server connectivity
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "allenai/Molmo-7B-D-0924", "messages": [{"role": "user", "content": "test"}]}'
```

**4. Memory Issues on RTX 3090:**
- Close other GPU applications
- Use `--gpu-memory-utilization 0.35` or lower
- Monitor with: `watch -n 1 nvidia-smi`

## 📈 Advanced Usage

### Analyzing Specific Log File
```bash
python3 -c "
from exp.helperFunc.training_analytics.generate_report import generate
generate(specific_log='exp/helperFunc/logs/my_custom_log.json')
"
```

### Batch Analysis of Multiple Runs
```python
import glob
from exp.helperFunc.training_analytics.generate_report import generate

for log_file in glob.glob('exp/helperFunc/logs/*.json'):
    generate(specific_log=log_file)
```

### Integration with Training Pipeline
Add automatic report generation at the end of training:

```python
from pytorch_lightning.callbacks import Callback
import subprocess

class AutoAnalyzeCallback(Callback):
    def on_fit_end(self, trainer, pl_module):
        # Generate report after training completes
        subprocess.run([
            'python3', '-m', 'exp.helperFunc.training_analytics.generate_report'
        ])

# Add to trainer
trainer = pl.Trainer(callbacks=[DebugLogger(), AutoAnalyzeCallback()])
```

## 🔗 Dependencies

```txt
# Core
pytorch-lightning>=2.0.0
torch>=2.0.0

# Analytics
scikit-learn>=1.3.0
numpy>=1.24.0
Pillow>=10.0.0

# AI Reasoning
vllm>=0.3.0
transformers>=4.35.0
bitsandbytes>=0.41.0  # For 4-bit quantization

# Optional
tensorflow>=2.13.0  # For Molmo preprocessing
```

## 📄 License & Attribution

**Author:** Vision Dev Project  
**Repository:** https://github.com/aniketqw/vision_dev_project  
**Last Updated:** January 2024

This package integrates:
- PyTorch Lightning for training orchestration
- vLLM for high-performance inference
- AllenAI's Molmo-7B-D for multimodal reasoning
- Custom analytics for deep training insights

---

**Pro Tip:** For production use, consider running the vLLM server on a dedicated GPU instance and update the `VLLM_URL` in the report generator to point to your production endpoint.
```

This README.md file is ready to copy-paste into your project. It includes:
1. Clear file structure
2. Step-by-step usage instructions
3. Configuration options
4. Troubleshooting guide
5. Advanced usage patterns
6. Proper markdown formatting for GitHub display