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

    samples = data.get("misclassified_samples", [])
    summary = data.get("summary", {})
    report_content = []

    for i, sample in enumerate(samples[:10]):
        img_data = sample.get('image_base64')
        if not img_data:
            logger.info(f"⚠️ Sample {i} missing image data. Skipping.")
            continue

        # FIX: Added <|image|> token. This is MANDATORY for Molmo to process visual data.
        # We also updated the prompt to be more direct about visual evidence.
        prompt = (
            f"<|image|>\n"
            f"This is a Machine Learning classification failure. "
            f"Ground Truth: {sample['true_label']}, Prediction: {sample['predicted_label']}. "
            f"Analyze the specific visual features (textures, shapes, or lighting) in this image. "
            f"Why did the model confuse these two classes? Give a concise, evidence-based reason."
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
            "temperature": 0.1, # Lowered for strictly factual analysis
            "max_tokens": 350
        }

        try:
            logger.info(f"🔄 Processing Sample {i+1}/{len(samples[:10])}...")
            res = requests.post(VLLM_URL, json=payload, timeout=90)
            res.raise_for_status()
            
            reasoning = res.json()['choices'][0]['message']['content']
            
            report_content.append(f"=== FAILURE CASE {i+1} ===\n")
            report_content.append(f"GROUND TRUTH: {sample['true_label']}\n")
            report_content.append(f"PREDICTED:    {sample['predicted_label']}\n")
            report_content.append(f"AI DIAGNOSIS: {reasoning}\n")
            report_content.append("-" * 30 + "\n\n")
            
        except Exception as e:
            logger.info(f"❌ Error analyzing sample {i}: {e}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = os.path.join(REPORT_OUTPUT_DIR, f"molmo_analysis_{timestamp}.txt")
    
    with open(out_file, 'w') as f:
        f.write(f"VISION-DEV AI REASONING REPORT\nGenerated: {datetime.now()}\n")
        f.write("=" * 40 + "\n\n")
        f.write("".join(report_content))
    
    logger.info(f"✅ Reasoning report successfully saved to: {out_file}")

if __name__ == "__main__":
    generate()