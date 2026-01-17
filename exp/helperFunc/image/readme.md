# Important


### Ensure you are in your venv
source venv_vision/bin/activate

### Start the server (this will take a few minutes to download the first time)
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --trust-remote-code \
    --gpu-memory-utilization 0.5 \
    --max-model-len 4096 \
    --port 8000
### so that it is gpu-memory-utilization 0.5 is within 14.91GiB
sudo renice -n 15 -p 271230

 sudo nice -n -10 python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-VL-7B-Instruct --quantization bitsandbytes --gpu-memory-utilization 0.4 --port 8000


 ### create a dedicated space for model
 mkdir -p /mnt/data/pratik_models

 Give your user ownership so Hugging Face can write there
sudo chown -R pratik2:pratik2 /mnt/data/pratik_models




### Step 2: The Permanent Fix (Moving your Project)
Since your home directory is on the small partition, you should move your entire work folder to the large drive and create a Symbolic Link (Symlink). This makes the system "think" the files are still in your home folder, but they are physically stored on the massive 1.9TB disk.

1. Move the project folder to the big drive
sudo mv /home/pratik2/vision_dev_project /mnt/data/

2. Give yourself ownership of the moved folder
sudo chown -R pratik2:pratik2 /mnt/data/vision_dev_project

 3. Create the Symlink in your home directory
ln -s /mnt/data/vision_dev_project /home/pratik2/vision_dev_project

### Launch vLLM using the Big Drive
When you run sudo, it often resets environment variables. To ensure vLLM downloads the model to /mnt/data and not back onto your full disk, we pass the environment variable directly inside the sudo command.

sudo nice -n -10 env HUGGINGFACE_HUB_CACHE="/mnt/data/pratik_models" \
/home/pratik2/vision_dev_project/venv_vision/bin/python3 -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen2.5-VL-7B-Instruct \
--quantization bitsandbytes \
--gpu-memory-utilization 0.3 \
--max-model-len 4096 \
--port 8000
 * or this
sudo renice -n 15 -p 271230
python3 -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen2.5-VL-7B-Instruct \
--quantization bitsandbytes \
--gpu-memory-utilization 0.2 \
--max-model-len 1024 \
--max-num-seqs 1 \
--port 8000


* or this we need to use the more stable V0 engine and disable CUDA Graphs (which take up ~2-3GB of VRAM just for speed optimization)

(venv_vision) pratik2@rtx3090:~/vision_dev_project$ VLLM_USE_V1=0 env HUGGINGFACE_HUB_CACHE="/mnt/data/pratik_models" \
python3 -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen2.5-VL-7B-Instruct \
--quantization bitsandbytes \
--gpu-memory-utilization 0.4 \
--max-model-len 2048 \
--enforce-eager \
--port 8000


### for git 

git config --local user.name "aniketqw"
git config --local user.email "38223792+aniketqw@users.noreply.github.com"
* for 1 hour window we can push and pull without credential for 1hour
git config --local credential.helper 'cache --timeout=3600'

This dubious ownership error is a security feature of Git. It happens because you moved the project to /mnt/data/ (likely using sudo), which changed the owner of the files or the path to a location Git doesn't "trust" by default.

Since you are the owner of the project and this is your local machine, you just need to tell Git that this specific directory is safe.

🛠️ The Fix: Register the Safe Directory
Run this command exactly as suggested by the error message:

Bash

git config --global --add safe.directory /mnt/data/vision_dev_project
🔍 Why did this happen?
Git checks if the user running the command is the same as the user who owns the .git folder. When you moved the folder to /mnt/data/, the file system metadata changed. Git blocks access to prevent a different user from executing malicious code via your repository.