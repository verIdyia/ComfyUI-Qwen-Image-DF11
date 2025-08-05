# ComfyUI Qwen-Image DFloat11 Nodes

ComfyUI custom nodes for the DFloat11 compressed Qwen-Image model. This package provides efficient image generation with reduced memory usage through DFloat11 compression technology.

## Features

- **DFloat11 Qwen-Image Loader**: Load the compressed Qwen-Image model with CPU offloading options
- **Qwen-Image Text Encode**: Process text prompts with language-specific magic prompts
- **Qwen-Image Sampler**: Generate images with full control over parameters
- **Qwen-Image Decode**: Save generated images to output directory
- **Qwen-Image Aspect Ratio**: Quick aspect ratio selection for common formats
- **Qwen-Image Preset Sampler**: Fast generation with preset configurations

## Installation

### 1. Install Custom Nodes
Clone this repository into your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/verIdyia/ComfyUI-Qwen-Image-DF11.git
```

Install dependencies:
```bash
cd ComfyUI-Qwen-Image-DF11
pip install -r requirements.txt
```

### 2. Download Models
Download the Qwen-Image models and place them in your ComfyUI models folder:

**Method 1: Using Git LFS (Recommended)**
```bash
cd ComfyUI/models/diffusion_models
git lfs clone https://huggingface.co/Qwen/Qwen-Image
git lfs clone https://huggingface.co/DFloat11/Qwen-Image-DF11
```

**Method 2: Simple DFloat11 Only (Hybrid approach)**
Download just the DFloat11 compressed transformer:
```bash
cd ComfyUI/models/diffusion_models
mkdir Qwen-Image-DF11
cd Qwen-Image-DF11
# Download diffusion_pytorch_model.safetensors from DFloat11/Qwen-Image-DF11
```

**Method 3: Full Local Model Structure (Offline)**
Download and organize all components locally:
```
diffusion_models/
└── Qwen-Image/
    ├── transformer/
    │   ├── config.json
    │   └── diffusion_pytorch_model.safetensors (from DFloat11)
    ├── text_encoder/
    │   └── [text encoder files from Qwen/Qwen-Image]
    ├── vae/
    │   └── [VAE files from Qwen/Qwen-Image]
    ├── tokenizer/
    │   └── [tokenizer files from Qwen/Qwen-Image]
    ├── scheduler/
    │   └── [scheduler config from Qwen/Qwen-Image]
    └── model_index.json
```

**Important Notes:**
- **Method 2**: Uses DFloat11 compressed transformer + downloads other components online (VAE, text encoder, etc.)
- **Method 3**: Fully offline but requires downloading all components from the original Qwen-Image model
- The transformer is the largest component (~28GB), so using DFloat11 compression saves significant space

### 3. Restart ComfyUI
Restart ComfyUI to load the new nodes.

## Memory Requirements

- **Without CPU Offloading**: 32GB VRAM required
- **With CPU Offloading**: 16GB VRAM required
- Model Size: 28.42 GB (32% smaller than original BFloat16)

## Usage

See the included example workflow JSON file for a complete setup example.

## Model Information

This package uses the DFloat11 compressed version of Qwen-Image:
- Model: `DFloat11/Qwen-Image-DF11`
- Original: `Qwen/Qwen-Image`
- Compression: Lossless 32% size reduction
- Performance: Bit-identical outputs to original model

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

The underlying Qwen-Image model is also licensed under Apache 2.0. Please refer to the original model's license terms:
- [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) - Apache 2.0
- [DFloat11/Qwen-Image-DF11](https://huggingface.co/DFloat11/Qwen-Image-DF11) - Please check the model's license on Hugging Face