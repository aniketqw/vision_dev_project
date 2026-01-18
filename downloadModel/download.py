import os
from huggingface_hub import snapshot_download

# 1. Define models and their local folder names
models_to_get = {
    # "molmo-7b": "allenai/Molmo-7B-D-0924",
    "pixtral-12b": "mistralai/Pixtral-12B-2409",
    "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
    "deepseek-llm-7b": "deepseek-ai/deepseek-coder-7b-instruct-v1.5"  # Better 7B model
}

# 2. Create the parent folder
os.makedirs("models", exist_ok=True)
os.chdir("models")

# 3. Download each model
for folder_name, repo_id in models_to_get.items():
    print(f"\n🚀 Downloading {repo_id} into models/{folder_name}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=folder_name,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✅ Successfully downloaded {repo_id}")
    except Exception as e:
        print(f"❌ Failed to download {repo_id}: {e}")

# 4. Return to project root
os.chdir("..")

# 5. Generate the README.md instructions
readme_content = """# 🧠 Local Model Library (RTX 3090 / 24GB Optimized)

This folder contains the local weights for your vision and reasoning models. 
To run these, ensure your `venv_vision` is active and you are in the project root.

## Important Setup Notes:

1. **Install vLLM with vision support first:**
```bash
pip install vllm torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install "vllm[vision]"
```

2. **For 4-bit quantization support (DeepSeek models):**
```bash
pip install bitsandbytes
```

## Model Commands:

### 1. Pixtral 12B (Best Modern Vision)
**Use for:** Complex scene reasoning and high-detail image analysis.

```bash
vllm serve ./models/pixtral-12b \\
  --tokenizer-mode mistral \\
  --limit-mm-per-prompt image=2 \\
  --max-model-len 8192 \\
  --gpu-memory-utilization 0.9 \\
  --port 8000
```

### 2. DeepSeek-Coder 7B (Best Reasoning/Coding - Fits 24GB)
**Use for:** Complex logic and programming. (Text only)

```bash
vllm serve ./models/deepseek-llm-7b \\
  --gpu-memory-utilization 0.85 \\
  --max-model-len 4096 \\
  --port 8001
```

### 3. Qwen2-VL 7B (Best OCR/Document Reading)
**Use for:** Reading small text and dense documents.

```bash
vllm serve ./models/qwen2-vl-7b \\
  --max-model-len 8192 \\
  --gpu-memory-utilization 0.7 \\
  --port 8002
```

### 4. Molmo 7B (Best for Pointing/Grounding)
**Use for:** Identifying exact coordinates of objects in an image.

```bash
vllm serve ./models/molmo-7b \\
  --trust-remote-code \\
  --gpu-memory-utilization 0.8 \\
  --max-model-len 2048 \\
  --port 8003
```

## Alternative: Ollama Setup (Easier)

If vLLM setup is complex, use Ollama instead:

```bash
# First, install Ollama: https://ollama.com/download
# Then pull and run models:

# Pixtral (when available in Ollama)
ollama run pixtral

# Qwen2-VL
ollama run qwen2-vl

# DeepSeek (7B fits, 70B needs quantization)
ollama run deepseek-coder:7b
```

## Monitoring

```bash
# Monitor VRAM usage
watch -n 1 nvidia-smi

# Check if API is running
curl http://localhost:8000/v1/models
```

## Test Requests

Save this as `test_request.py`:

```python
import requests
import json

# For Pixtral or Qwen2-VL
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "pixtral-12b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500
    }
)

print(json.dumps(response.json(), indent=2))
```

## Troubleshooting

**Out of Memory Errors:**
- Reduce `--gpu-memory-utilization` to 0.7 or lower
- Reduce `--max-model-len` to 2048 or 4096
- Close other GPU applications

**Model Won't Load:**
- Verify download completed: `ls -lh models/[model-name]`
- Check CUDA version: `nvidia-smi`
- Try running with `--dtype float16` flag

**API Connection Issues:**
- Verify port isn't in use: `lsof -i :8000`
- Check firewall settings
- Use `0.0.0.0` instead of `localhost` if accessing remotely
"""

# Write the README
with open("models/README.md", "w") as f:
    f.write(readme_content)

print("\n✅ All downloads complete!")
print("📄 README.md has been created in the models/ folder")
print("\n💡 Next steps:")
print("   1. Activate your venv: source venv_vision/bin/activate")
print("   2. Install vLLM: pip install vllm")
print("   3. Read models/README.md for usage instructions")