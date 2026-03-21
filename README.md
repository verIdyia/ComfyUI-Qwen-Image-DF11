# ComfyUI Qwen-Image DFloat11 Nodes

ComfyUI custom nodes for the **DFloat11 compressed Qwen-Image** model. Generate high-quality images with 32% less memory through lossless DFloat11 compression — bit-identical outputs to the original model.

## Nodes

| Node | Description |
|------|-------------|
| **DFloat11 Qwen-Image Loader** | Load the compressed model with optional CPU offloading |
| **Qwen-Image Text Encode** | Process text prompts with optional quality-boosting magic prompt |
| **Qwen-Image Sampler** | Generate images with full parameter control (steps, CFG, size, batch) |
| **Qwen-Image Aspect Ratio** | Quick preset aspect ratios (1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3, 21:9) |
| **Qwen-Image Preset (All-in-One)** | Single-node generation with quality presets (draft/fast/balanced/quality/max) |

## Installation

### 1. Install Custom Nodes

**Via ComfyUI Manager (recommended):**
Search for "Qwen-Image-DF11" in ComfyUI Manager and install.

**Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/verIdyia/ComfyUI-Qwen-Image-DF11.git
cd ComfyUI-Qwen-Image-DF11
pip install -r requirements.txt
```

### 2. Download Models

Set up the model directory structure in `ComfyUI/models/diffusion_models/`:

```
Qwen-Image/
├── model_index.json              # From Qwen/Qwen-Image
├── transformer/
│   ├── config.json               # From Qwen/Qwen-Image
│   └── diffusion_pytorch_model.safetensors  # From DFloat11/Qwen-Image-DF11
├── text_encoder/                 # From Qwen/Qwen-Image
├── vae/                          # From Qwen/Qwen-Image
├── tokenizer/                    # From Qwen/Qwen-Image
└── scheduler/                    # From Qwen/Qwen-Image
```

**Step 1 — Download the base model:**
```bash
cd ComfyUI/models/diffusion_models
git lfs clone https://huggingface.co/Qwen/Qwen-Image
```

**Step 2 — Replace the transformer with the DFloat11 version:**
```bash
cd Qwen-Image/transformer
mv diffusion_pytorch_model.safetensors diffusion_pytorch_model.safetensors.backup
wget https://huggingface.co/DFloat11/Qwen-Image-DF11/resolve/main/diffusion_pytorch_model.safetensors
```

**Step 3 — Merge config files** (required for DFloat11 loading):
```python
import json
from huggingface_hub import hf_hub_download

# Download DFloat11 config
df11_config_path = hf_hub_download("DFloat11/Qwen-Image-DF11", "config.json")
with open(df11_config_path) as f:
    df11_config = json.load(f)

# Merge with existing Qwen transformer config
config_path = "ComfyUI/models/diffusion_models/Qwen-Image/transformer/config.json"
with open(config_path) as f:
    qwen_config = json.load(f)

merged = {**qwen_config, **df11_config}
with open(config_path, "w") as f:
    json.dump(merged, f, indent=2)
```

### 3. Restart ComfyUI

## Memory Requirements

### VRAM (GPU)

| Mode | VRAM Required | Peak VRAM |
|------|--------------|-----------|
| CPU Offload ON | 16 GB+ | ~16.7 GB |
| CPU Offload OFF | 32 GB+ | ~29.7 GB |

### System RAM

| | Minimum | Recommended |
|---|---------|-------------|
| **RAM** | **96 GB** | **128 GB** |

> **Important:** During model loading, the system needs ~68 GB of RAM simultaneously (41 GB model shell + 27 GB compressed weights). 64 GB is insufficient and will crash on Windows. 96 GB works with headroom; 128 GB is ideal. Linux with 64 GB may work due to mmap overcommit, but is not guaranteed.

- Qwen-Image: 20.4B parameter model (bf16 = 41 GB)
- DFloat11 compressed transformer: **28.42 GB** (32% smaller than original)
- Text encoder (Qwen2.5-VL-7B): ~16 GB additional
- Lossless compression — produces bit-identical outputs

## Preset Quality Modes

| Preset | Steps | CFG | Use Case |
|--------|-------|-----|----------|
| draft | 12 | 3.0 | Quick previews |
| fast | 20 | 3.5 | Rapid iteration |
| balanced | 50 | 4.0 | General use |
| quality | 80 | 4.5 | High quality |
| max_quality | 100 | 5.0 | Best possible output |

## Requirements

- **System RAM**: 64 GB minimum, 128 GB recommended
- **GPU**: NVIDIA with 16+ GB VRAM (32+ GB without CPU offload)
- **CUDA**: 12.1+
- **Python**: 3.10+
- **Disk**: ~43 GB for model files
- diffusers >= 0.35.0, transformers >= 4.51.3, dfloat11[cuda12]

## License

Apache License 2.0 — see [LICENSE](LICENSE).

- [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) — Apache 2.0
- [DFloat11/Qwen-Image-DF11](https://huggingface.co/DFloat11/Qwen-Image-DF11) — See model license
