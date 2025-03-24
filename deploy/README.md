
# Doraemon Classifier Deployment API

## File Structure
The Doraemon model requires the following three files:
- `config.json` - Model configuration file
- `doraemon_modeling.py` - Model implementation code
- `xxx.pt` - Model weights file

## Loading the Model from Hugging Face
Here's an example of loading and using the Doraemon model with the Transformers library:

```python
from transformers import AutoModel, AutoProcessor
import requests
from io import BytesIO
from PIL import Image

# Load model and processor
pretrained_model_name_or_path = "user/repo"
model = AutoModel.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, revision="v2")
processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, revision="v2")

# Prepare images
urls = [
    "https://github.com/ultralytics/ultralytics/blob/main/ultralytics/assets/bus.jpg",
    "https://github.com/ultralytics/ultralytics/blob/main/ultralytics/assets/zidane.jpg",
]

images = []
for url in urls:
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    images.append(image)

# Process input
batch_inputs = processor(images, return_tensors="pt")

# Get output
probs = model(batch_inputs["pixel_values"])
print("Probs:\n", probs)
output = processor.postprocess(probs)
print("Tagging:")
for r in output:
    print(r)
```

## Usage

### Local API

#### Option A: Using Config File
Define `model_path` in `config.json` pointing to the absolute path of the model weights file.

#### Option B: Using Command Line
```bash
python doraemon_modeling.py --model_path /path/to/xxx.pt
```
This will override the settings in `config.json`.

### HF Remote API

#### Step 1: Create a Hugging Face Repository
```bash
# Install Hugging Face Hub
pip install huggingface_hub

# Login for authentication
huggingface-cli login
```

#### Step 2: Configure the Model
Set `model_path` in `config.json` with the format `user/repo/filename:revision`

Required components:
- `user`: Your Hugging Face username
- `repo`: Repository name
- `filename`: Weights file name (e.g., `best.pt`)
- `revision`: Version number (e.g., `v3`)

#### Step 3: Push to Hugging Face
Push the following files to your Hugging Face repository:
- `config.json`
- `doraemon_modeling.py`
- `xxx.pt`

#### Step 4: Call API
```bash
python doraemon_modeling.py --pretrained_model_name_or_path user/repo --revision v3
```

> **Note**: When `--pretrained_model_name_or_path` points to a Hugging Face repository, the local `deploy/doraemon_modeling.py` content will be ignored as the code from the repository will be executed instead.

## Version Control
Make sure to specify the correct `revision` parameter when loading the model to match the version of your deployed model.
