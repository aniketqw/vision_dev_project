import json
import os
import requests
import glob
from datetime import datetime
from logger_setup import logger

# --- CONFIGURATION ---
LOG_DIR = "exp/helperFunc/logs" 
REPORT_OUTPUT_DIR = "/home/pratik2/vision_dev_project/exp/helperFunc/training_analytics/data/vision_dev_reports"
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "allenai/Molmo-7B-D-0924"

os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

def get_latest_log():
    list_of_files = glob.glob(f'{LOG_DIR}/*.json')
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def generate():
    latest_log = get_latest_log()
    if not latest_log:
        logger.info(f"❌ No training logs found in {LOG_DIR}")
        return

    logger.info(f"📄 Analyzing latest log: {latest_log}")
    with open(latest_log, 'r') as f:
        data = json.load(f)

    # --- RUNTIME DATASET DISCOVERY ---
    summary = data.get("summary", {})
    dataset_info = summary.get("dataset_info", {})
    classes = dataset_info.get("classes", {})
    resolution = dataset_info.get("resolution", "32x32")
    
    samples = data.get("misclassified_samples", [])
    report_content = []

    for i, sample in enumerate(samples[:10]):
        img_data = sample.get('image_base64')
        if not img_data:
            logger.info(f"⚠️ Sample {i} missing image data. Skipping.")
            continue

        # Resolve labels to names using discovered metadata
        true_idx = str(sample['true_label'])
        pred_idx = str(sample['predicted_label'])
        true_name = classes.get(true_idx, f"Class {true_idx}")
        pred_name = classes.get(pred_idx, f"Class {pred_idx}")

        # The prompt is now fully dynamic and dataset-aware
        prompt = (
            f"<|image|>\n"
            f"You are a Senior AI Diagnostic tool analyzing {resolution} image classification failures.\n\n"
            f"CONTEXT:\n"
            f"- Ground Truth: {true_name} (Label {true_idx})\n"
            f"- Prediction: {pred_name} (Label {pred_idx})\n\n"
            f"TASK:\n"
            f"Analyze the visual features in this {resolution} image. Why did the model confuse "
            f"the '{true_name}' for a '{pred_name}'? Focus on pixel-level evidence like "
            f"silhouettes, color blobs, or background noise that might cause confusion at this low resolution."
        )

        payload = {
            "model": MODEL_NAME,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/png;base64,{img_data}"}
                    }
                ]
            }],
            "temperature": 0.1,
            "max_tokens": 350
        }

        try:
            logger.info(f"🔄 Processing Sample {i+1} ({true_name} vs {pred_name})...")
            res = requests.post(VLLM_URL, json=payload, timeout=90)
            res.raise_for_status()
            
            reasoning = res.json()['choices'][0]['message']['content']
            
            report_content.append(f"=== FAILURE CASE {i+1} ===\n")
            report_content.append(f"GROUND TRUTH: {true_name} ({true_idx})\n")
            report_content.append(f"PREDICTED:    {pred_name} ({pred_idx})\n")
            report_content.append(f"AI DIAGNOSIS: {reasoning}\n")
            report_content.append("-" * 30 + "\n\n")
            
        except Exception as e:
            logger.info(f"❌ Error analyzing sample {i}: {e}")

    # Save output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = os.path.join(REPORT_OUTPUT_DIR, f"ai_reasoning_{timestamp}.txt")
    
    with open(out_file, 'w') as f:
        f.write(f"VISION-DEV AI REASONING REPORT\n")
        f.write(f"Source Log: {os.path.basename(latest_log)}\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("=" * 40 + "\n\n")
        f.write("".join(report_content))
    
    logger.info(f"✅ Reasoning report successfully saved to: {out_file}")

if __name__ == "__main__":
    generate()