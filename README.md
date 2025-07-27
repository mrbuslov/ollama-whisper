## ⚠️ Read it first!
If you want a reliable and fast way to run Whisper without Ollama, you should go directly to this section - [Without Ollama](#without-ollama).  
However, if you still want to play around with it and try deploying it with Ollama, you can go to this section - [With Ollama](#run-ollama) - but in my experience, it’s not reliable. The support will probably come in the future - https://github.com/ollama/ollama/issues/1168  
Personally, no matter where I ran it - on CPU or GPU - I encountered the following errors.
```
time=2025-07-27T14:39:19.830Z level=INFO source=routes.go:1235 msg="server config" env="map[CUDA_VISIBLE_DEVICES: GPU_DEVICE_ORDINAL: HIP_VISIBLE_DEVICES: HSA_OVERRIDE_GFX_VERSION: HTTPS_PROXY: HTTP_PROXY: NO_PROXY: OLLAMA_CONTEXT_LENGTH:4096 OLLAMA_DEBUG:INFO OLLAMA_FLASH_ATTENTION:false OLLAMA_GPU_OVERHEAD:0 OLLAMA_HOST:http://0.0.0.0:11434 OLLAMA_INTEL_GPU:false OLLAMA_KEEP_ALIVE:5m0s OLLAMA_KV_CACHE_TYPE: OLLAMA_LLM_LIBRARY: OLLAMA_LOAD_TIMEOUT:5m0s OLLAMA_MAX_LOADED_MODELS:0 OLLAMA_MAX_QUEUE:512 OLLAMA_MODELS:/root/.ollama/models OLLAMA_MULTIUSER_CACHE:false OLLAMA_NEW_ENGINE:false OLLAMA_NOHISTORY:false OLLAMA_NOPRUNE:false OLLAMA_NUM_PARALLEL:0 OLLAMA_ORIGINS:[http://localhost https://localhost http://localhost:* https://localhost:* http://127.0.0.1 https://127.0.0.1 http://127.0.0.1:* https://127.0.0.1:* http://0.0.0.0 https://0.0.0.0 http://0.0.0.0:* https://0.0.0.0:* app://* file://* tauri://* vscode-webview://* vscode-file://*] OLLAMA_SCHED_SPREAD:false ROCR_VISIBLE_DEVICES: http_proxy: https_proxy: no_proxy:]"
time=2025-07-27T14:39:19.831Z level=INFO source=images.go:476 msg="total blobs: 4"
time=2025-07-27T14:39:19.831Z level=INFO source=images.go:483 msg="total unused blobs removed: 0"
time=2025-07-27T14:39:19.831Z level=INFO source=routes.go:1288 msg="Listening on [::]:11434 (version 0.9.6)"
time=2025-07-27T14:39:19.831Z level=INFO source=gpu.go:217 msg="looking for compatible GPUs"
time=2025-07-27T14:39:19.833Z level=INFO source=gpu.go:377 msg="no compatible GPUs were discovered"
�time=2025-07-27T14:39:19.833Z level=INFO source=types.go:130 msg="inference compute" id=0 library=cpu variant="" compute="" driver=0.0 name="" total="15.3 GiB" available="6.5 GiB"
T[GIN] 2025/07/27 - 14:40:09 | 200 |      46.517µs |       127.0.0.1 | HEAD     "/"
\[GIN] 2025/07/27 - 14:40:09 | 200 |     489.956µs |       127.0.0.1 | GET      "/api/tags"
T[GIN] 2025/07/27 - 14:40:15 | 200 |      18.122µs |       127.0.0.1 | HEAD     "/"
Z[GIN] 2025/07/27 - 14:40:15 | 200 |      70.749µs |       127.0.0.1 | GET      "/api/ps"
T[GIN] 2025/07/27 - 14:40:31 | 200 |      23.103µs |       127.0.0.1 | HEAD     "/"
[GIN] 2025/07/27 - 14:40:31 | 200 |    2.520231ms |       127.0.0.1 | POST     "/api/show"
�time=2025-07-27T14:40:31.534Z level=INFO source=server.go:135 msg="system memory" total="15.3 GiB" free="6.7 GiB" freeswap="1.7 GiB"
time=2025-07-27T14:40:31.534Z level=WARN source=memory.go:129 msg="model missing blk.0 layer size"
panic: interface conversion: interface {} is nil, not *ggml.array[string]
goroutine 36 [running]:
�github.com/ollama/ollama/fs/ggml.GGML.GraphSize({{0x60cd13cb1630, 0xc00044ef50}, {0x60cd13cb15e0, 0xc00018d808}, 0x351bb8e0}, 0x2000, 0x200, 0x2, {0x0, 0x0})
        github.com/ollama/ollama/fs/ggml/ggml.go:481 +0x1614
github.com/ollama/ollama/llm.EstimateGPULayers({, *, *}, , {, *, }, {{0x2000, 0x200, 0xffffffffffffffff, ...}, ...}, ...)
        github.com/ollama/ollama/llm/memory.go:142 +0x725
�github.com/ollama/ollama/llm.NewLlamaServer({0xc00046c0c0, 0x1, 0x1}, {0xc0001748c0, 0x62}, 0xc000408030, {0x0, 0x0, _}, {0x0, ...}, ...)
        github.com/ollama/ollama/llm/server.go:149 +0x4f0
github.com/ollama/ollama/server.(Scheduler).load(0xc00009d440, 0xc000020d00, 0xc000408030, {0xc00046c0c0, 0x1, 0x1}, 0x2)
        github.com/ollama/ollama/server/sched.go:447 +0x19d
github.com/ollama/ollama/server.(Scheduler).processPending(0xc00009d440, {0x60cd13cb5700, 0xc0000b6e60})
        github.com/ollama/ollama/server/sched.go:216 +0xcc5
github.com/ollama/ollama/server.(Scheduler).Run.func1()
        github.com/ollama/ollama/server/sched.go:110 +0x1f
created by github.com/ollama/ollama/server.(*Scheduler).Run in goroutine 1
        github.com/ollama/ollama/server/sched.go:109 +0xb1
```
These issues are not related to the hardware, but rather to an incompatibility between Ollama and Whisper. This is because Ollama was originally designed to **support only LLMs**, and it still only officially supports them. It does not support models that work with audio, even though it publicly lists the latest audio models in its interface.

## Run Ollama
### Preferrable way - docker:
```
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama:/root/.ollama \
  ollama/ollama
```

### Using system

- Linux/Mac  
```curl -fsSL https://ollama.ai/install.sh | sh```
- Windows - download from https://ollama.ai/

Stop automatically Ollama start
If you don’t want Ollama to launch on boot or respawn automatically:
```
sudo systemctl disable ollama.service
```

### Models:
- `ZimaBlueAI/whisper-large-v3`         # main model
- `dimavz/whisper-tiny`                 # light version

### Run Ollama Container
Enter the container shell
```docker exec -it ollama bash```
List currently installed models
```ollama list```
Pull the model
```ollama pull ZimaBlueAI/whisper-large-v3```
Confirm and run
```
ollama list
ollama run ZimaBlueAI/whisper-large-v3
```
### After using
You can stop model now
```
ollama stop ZimaBlueAI/whisper-large-v3
```

Check if it was stopped
```
ollama list
```


## Convert audio file
### Without Ollama
- Make venv and activate to prevent all of the rubbish put into your system:
    - `python -m venv venv`
    - `. venv/bin/activate`
- Run `pip install openai-whisper` - NOTE: this will take a while
- Run the code below. Create file `whisper_test.py`, paste the code below, run with `python whisper_test.py`
```python
# pip install openai-whisper
import whisper
import os

"""
Available Models:

tiny (~39 MB) - fastest but less accurate
base (~74 MB) - fast with basic accuracy
small (~244 MB) - medium speed and accuracy
medium (~769 MB) - good accuracy
large-v3 (~1550 MB) - best accuracy
"""

MODEL_CACHE_DIR = "./whisper_models"  # Change this path as needed
MODEL_NAME = "small"  # Available: tiny, base, small, medium, large-v3
AUDIO_FILE = "test_audio.mp3"  # NOTE: adjust it if needed

def setup_model_cache():
    """Creates model cache directory if it doesn't exist"""
    if not os.path.exists(MODEL_CACHE_DIR):
        os.makedirs(MODEL_CACHE_DIR)
        print(f"Created models directory: {MODEL_CACHE_DIR}")


def load_whisper_model(model_name=MODEL_NAME):
    """
    Loads Whisper model and saves it to the specified directory
    """
    setup_model_cache()

    # Set environment variable for custom cache path
    os.environ["WHISPER_CACHE_DIR"] = MODEL_CACHE_DIR

    print(f"Loading model: {model_name}")
    print(f"Save path: {MODEL_CACHE_DIR}")

    try:
        # Load model with custom cache path
        model = whisper.load_model(model_name, download_root=MODEL_CACHE_DIR)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def transcribe_audio(audio_file_path, model_name=MODEL_NAME, language=None):
    """
    Transcribes audio file to text

    Args:
        audio_file_path (str): Path to audio file
        model_name (str): Whisper model name
        language (str, optional): Language for transcription (e.g., 'ru', 'en')

    Returns:
        str: Transcribed text
    """
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    # Load model
    model = load_whisper_model(model_name)
    if model is None:
        return None

    print(f"Transcribing file: {audio_file_path}")

    try:
        # Perform transcription
        options = {}
        if language:
            options["language"] = language

        result = model.transcribe(audio_file_path, **options)

        print("Transcription completed!")
        return result["text"]

    except Exception as e:
        print(f"Transcription error: {e}")
        return None


def get_model_info():
    """Displays information about available models"""
    models_info = {
        "tiny": "~39 MB, fastest but less accurate",
        "base": "~74 MB, fast with basic accuracy",
        "small": "~244 MB, medium speed and accuracy",
        "medium": "~769 MB, good accuracy",
        "large-v3": "~1550 MB, best accuracy",
    }

    print("Available Whisper models:")
    for model, description in models_info.items():
        print(f"  {model}: {description}")


def main():
    """Main function for usage example"""
    # Show model information
    get_model_info()

    # Path to your audio file
    audio_file = "path/to/your/audio.mp3"  # Change to actual path

    # Check if file exists
    if not os.path.exists(AUDIO_FILE):
        print(f"File not found: {AUDIO_FILE}")
        print("Please specify the correct path to the audio file")
        return

    # Transcribe audio
    text = transcribe_audio(
        audio_file_path=AUDIO_FILE,
        model_name=MODEL_NAME,
        language="en",  # Specify English language, remove for auto-detection
    )

    if text:
        print("\n" + "=" * 50)
        print("TRANSCRIPTION RESULT:")
        print("=" * 50)
        print(text)
    else:
        print("Failed to perform transcription")


if __name__ == "__main__":
    main()
```


### With Ollama
```python
import requests
import json
import base64

MODEL_NAME = 'ZimaBlueAI/whisper-large-v3'  # NOTE: adjust this if required 
OLLAMA_BASE_URL = 'http://localhost:11434'  # NOTE: adjust this if required 

def transcribe_audio_ollama(audio_file_path, model_name=MODEL_NAME):
    # Read audio
    with open(audio_file_path, "rb") as audio_file:
        audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
    
    # send to Ollama
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": "",
        "stream": False,
        "images": [audio_data]  # Ollama can accept audio and image data
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    return result["response"]

file_path = "path/to/your/file.mp3"    # NOTE: adjust this
text = transcribe_audio_ollama(file_path)
print(text)
```
