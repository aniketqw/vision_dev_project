import json
import os
import requests
import glob
from datetime import datetime
from logger_setup import logger
# --- CONFIGURATION ---
# Based on your project structure: ~/vision_dev_project/exp/helperFunc/logs
LOG_DIR = "exp/helperFunc/logs" 
REPORT_OUTPUT_DIR = "/home/pratik2/vision_dev_project/exp/helperFunc/training_analytics/data/vision_dev_reports"
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "allenai/Molmo-7B-D-0924"

os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

def get_latest_log():
    list_of_files = glob.glob(f'{LOG_DIR}/*.json')
    if not list_of_files:
        return None
    # Returns the most recently created file
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

    # Limit to top 10 samples to manage token limits and time
    for i, sample in enumerate(samples[:10]):
        # FIX: Changed key from 'image' to 'image_base64' to match your JSON
        img_data = sample.get('image_base64')
        if not img_data:
            logger.info(f"⚠️ Sample {i} missing image data. Skipping.")
            continue

        prompt = f"""
        Analyze this Machine Learning classification failure:
        - True Label (GT): {sample['true_label']}
        - Predicted Label: {sample['predicted_label']}
        - Training Epoch: {sample.get('epoch', 'N/A')}
        - Final Validation Loss: {summary.get('final_val_loss', 'N/A')}
        
        Task: Look at the visual features of the provided image. Provide a concise technical 
        reason why the model confused class {sample['true_label']} for class {sample['predicted_label']}.
        """

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
            "temperature": 0.2,
            "max_tokens": 300
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

    # Save the final analysis
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = os.path.join(REPORT_OUTPUT_DIR, f"molmo_analysis_{timestamp}.txt")
    
    with open(out_file, 'w') as f:
        f.write(f"VISION-DEV AI REASONING REPORT\nGenerated: {datetime.now()}\n")
        f.write("=" * 40 + "\n\n")
        f.write("".join(report_content))
    
    logger.info(f"✅ Reasoning report successfully saved to: {out_file}")

if __name__ == "__main__":
    generate()