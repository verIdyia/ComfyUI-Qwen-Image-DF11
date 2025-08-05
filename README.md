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

1. Clone this repository into your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-Qwen-Image-DF11.git
```

2. Install dependencies:
```bash
cd ComfyUI-Qwen-Image-DF11
pip install -r requirements.txt
```

3. Restart ComfyUI

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

See the original Qwen-Image model license terms.