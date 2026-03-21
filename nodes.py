import torch
import numpy as np
import os
import json
import logging

from PIL import Image
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from transformers.modeling_utils import no_init_weights

import folder_paths
import comfy.model_management
import comfy.utils

logger = logging.getLogger("ComfyUI-Qwen-Image-DF11")


def _get_device():
    return comfy.model_management.get_torch_device()


def _get_offload_device():
    return comfy.model_management.unet_offload_device()


def _find_qwen_models():
    """Scan diffusion_models paths for directories containing Qwen-Image models."""
    qwen_models = []
    for path in folder_paths.get_folder_paths("diffusion_models"):
        if not os.path.exists(path):
            continue
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path) and "qwen" in item.lower():
                # Quick validation: check for model_index.json
                if os.path.exists(os.path.join(item_path, "model_index.json")):
                    qwen_models.append(item)
    return qwen_models


def _resolve_model_path(model_name):
    """Resolve model name to full path."""
    for path in folder_paths.get_folder_paths("diffusion_models"):
        candidate = os.path.join(path, model_name)
        if os.path.exists(candidate):
            return candidate
    raise ValueError(f"Model not found: {model_name}")


def _validate_model_structure(model_path):
    """Validate that all required components exist."""
    required = {
        "transformer": "transformer/config.json",
        "vae": "vae",
        "text_encoder": "text_encoder",
        "tokenizer": "tokenizer",
        "scheduler": "scheduler",
    }
    missing = []
    for name, subpath in required.items():
        if not os.path.exists(os.path.join(model_path, subpath)):
            missing.append(name)
    if missing:
        raise ValueError(
            f"Missing components in {model_path}: {missing}\n"
            "Download the complete Qwen-Image model structure. See README for instructions."
        )

    df11_path = os.path.join(model_path, "transformer", "diffusion_pytorch_model.safetensors")
    if not os.path.exists(df11_path):
        raise ValueError(
            f"DFloat11 compressed transformer not found at: {df11_path}\n"
            "Download from DFloat11/Qwen-Image-DF11 on Hugging Face."
        )


ASPECT_RATIOS = {
    "1:1 (1328x1328)": (1328, 1328),
    "16:9 (1664x928)": (1664, 928),
    "9:16 (928x1664)": (928, 1664),
    "4:3 (1472x1104)": (1472, 1104),
    "3:4 (1104x1472)": (1104, 1472),
    "3:2 (1536x1024)": (1536, 1024),
    "2:3 (1024x1536)": (1024, 1536),
    "21:9 (1680x720)": (1680, 720),
    "9:21 (720x1680)": (720, 1680),
}


class DFloat11QwenImageLoader:
    """Load DFloat11-compressed Qwen-Image model with optional CPU offloading."""

    _cached_pipe = None
    _cached_path = None
    _cached_offload = None

    @classmethod
    def INPUT_TYPES(cls):
        models = _find_qwen_models()
        if not models:
            models = ["(No Qwen models found - check diffusion_models folder)"]
        return {
            "required": {
                "model_name": (models, {"default": models[0]}),
                "cpu_offload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable CPU offloading to reduce VRAM usage (16GB VRAM). Disable for full GPU (32GB VRAM).",
                }),
                "pin_memory": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Pin CPU memory for faster CPU-GPU transfers. Uses more system RAM.",
                }),
            }
        }

    RETURN_TYPES = ("QWEN_PIPE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_model"
    CATEGORY = "Qwen-Image/DFloat11"
    DESCRIPTION = "Load DFloat11-compressed Qwen-Image model. 32% smaller than original with identical outputs."

    def load_model(self, model_name, cpu_offload, pin_memory):
        if model_name.startswith("(No Qwen"):
            raise ValueError(
                "No Qwen-Image model found.\n"
                "Place models in: ComfyUI/models/diffusion_models/\n"
                "See README for download instructions."
            )

        # Check system RAM - this model requires significant memory for loading
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / 1024**3
            if available_gb < 50:
                logger.warning(
                    f"Low system RAM detected ({available_gb:.0f}GB available). "
                    "DFloat11 Qwen-Image requires ~68GB RAM during loading. "
                    "Consider closing other applications or increasing virtual memory/swap."
                )
        except ImportError:
            pass

        # Return cached model if same config
        if (
            DFloat11QwenImageLoader._cached_pipe is not None
            and DFloat11QwenImageLoader._cached_path == model_name
            and DFloat11QwenImageLoader._cached_offload == cpu_offload
        ):
            logger.info("Using cached Qwen-Image pipeline")
            return (DFloat11QwenImageLoader._cached_pipe,)

        full_path = _resolve_model_path(model_name)
        _validate_model_structure(full_path)

        comfy.model_management.soft_empty_cache()

        pbar = comfy.utils.ProgressBar(3)

        # Step 1: Create transformer shell with DFloat11 weights
        logger.info("Loading DFloat11 compressed transformer...")
        with no_init_weights():
            transformer = QwenImageTransformer2DModel.from_config(
                QwenImageTransformer2DModel.load_config(
                    full_path, subfolder="transformer",
                ),
            ).to(torch.bfloat16)

        from dfloat11 import DFloat11Model
        DFloat11Model.from_pretrained(
            os.path.join(full_path, "transformer"),
            device="cpu",
            cpu_offload=cpu_offload,
            pin_memory=pin_memory,
            bfloat16_model=transformer,
        )
        pbar.update(1)

        # Step 2: Build full pipeline
        logger.info("Loading pipeline components (VAE, text encoder, tokenizer, scheduler)...")
        pipe = QwenImagePipeline.from_pretrained(
            full_path,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        pbar.update(1)

        # Step 3: Move to device
        if cpu_offload:
            logger.info("Enabling CPU offloading (16GB VRAM mode)")
            pipe.enable_model_cpu_offload()
        else:
            device = _get_device()
            logger.info(f"Moving pipeline to {device} (32GB VRAM mode)")
            pipe.to(device)
        pbar.update(1)

        # Cache the pipeline
        DFloat11QwenImageLoader._cached_pipe = pipe
        DFloat11QwenImageLoader._cached_path = model_name
        DFloat11QwenImageLoader._cached_offload = cpu_offload

        logger.info("Qwen-Image DFloat11 pipeline loaded successfully")
        return (pipe,)


class QwenImageTextEncode:
    """Encode text prompt with optional quality-boosting magic prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe the image you want to generate...",
                    "dynamicPrompts": True,
                }),
            },
            "optional": {
                "language": (["en", "zh"], {"default": "en"}),
                "magic_prompt": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Append quality-boosting suffix to prompt.",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "Qwen-Image/DFloat11"

    MAGIC_PROMPTS = {
        "en": "masterpiece, best quality, ultra detailed, 8K, RAW photo",
        "zh": "杰作，最高品质，超精细，8K，RAW照片",
    }

    def encode(self, text, language="en", magic_prompt=True):
        if magic_prompt and text.strip():
            positive = f"{text}, {self.MAGIC_PROMPTS[language]}"
        else:
            positive = text
        negative = ""
        return (positive, negative)


class QwenImageSampler:
    """Generate images using the Qwen-Image pipeline with full parameter control."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_PIPE",),
                "positive": ("STRING", {"forceInput": True}),
                "negative": ("STRING", {"forceInput": True}),
                "width": ("INT", {"default": 1328, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 1328, "min": 256, "max": 2048, "step": 16}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 150, "step": 1}),
                "cfg": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip": "True CFG scale. Higher = more prompt adherence.",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "Qwen-Image/DFloat11"

    def sample(self, pipeline, positive, negative, width, height, steps, cfg, seed, batch_size=1):
        device = _get_device()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        pbar = comfy.utils.ProgressBar(steps)

        def step_callback(pipe, step_index, timestep, callback_kwargs):
            pbar.update(1)
            return callback_kwargs

        comfy.model_management.soft_empty_cache()

        with torch.inference_mode():
            result = pipeline(
                prompt=positive,
                negative_prompt=negative if negative.strip() else None,
                width=width,
                height=height,
                num_inference_steps=steps,
                true_cfg_scale=cfg,
                num_images_per_prompt=batch_size,
                generator=generator,
                callback_on_step_end=step_callback,
            )

        # Convert PIL images to ComfyUI tensor format [B, H, W, C]
        images = []
        for img in result.images:
            img_np = np.array(img).astype(np.float32) / 255.0
            images.append(img_np)

        image_tensor = torch.from_numpy(np.stack(images))
        return (image_tensor,)


class QwenImageAspectRatio:
    """Select from preset aspect ratios optimized for Qwen-Image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (list(ASPECT_RATIOS.keys()), {"default": "1:1 (1328x1328)"}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_dimensions"
    CATEGORY = "Qwen-Image/DFloat11"

    def get_dimensions(self, aspect_ratio):
        width, height = ASPECT_RATIOS[aspect_ratio]
        return (width, height)


class QwenImagePresetSampler:
    """One-node image generation with preset quality configurations."""

    PRESETS = {
        "draft": {"steps": 12, "cfg": 3.0},
        "fast": {"steps": 20, "cfg": 3.5},
        "balanced": {"steps": 50, "cfg": 4.0},
        "quality": {"steps": 80, "cfg": 4.5},
        "max_quality": {"steps": 100, "cfg": 5.0},
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_PIPE",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe the image...",
                    "dynamicPrompts": True,
                }),
                "preset": (list(cls.PRESETS.keys()), {"default": "balanced"}),
                "aspect_ratio": (list(ASPECT_RATIOS.keys()), {"default": "1:1 (1328x1328)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "magic_prompt": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Qwen-Image/DFloat11"
    DESCRIPTION = "All-in-one node: text encode + sample in a single step."

    def generate(self, pipeline, text, preset, aspect_ratio, seed, magic_prompt=True):
        config = self.PRESETS[preset]
        width, height = ASPECT_RATIOS[aspect_ratio]

        if magic_prompt and text.strip():
            prompt = f"{text}, masterpiece, best quality, ultra detailed, 8K, RAW photo"
        else:
            prompt = text

        device = _get_device()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        pbar = comfy.utils.ProgressBar(config["steps"])

        def step_callback(pipe, step_index, timestep, callback_kwargs):
            pbar.update(1)
            return callback_kwargs

        comfy.model_management.soft_empty_cache()

        with torch.inference_mode():
            result = pipeline(
                prompt=prompt,
                negative_prompt=None,
                width=width,
                height=height,
                num_inference_steps=config["steps"],
                true_cfg_scale=config["cfg"],
                generator=generator,
                callback_on_step_end=step_callback,
            )

        image_np = np.array(result.images[0]).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        return (image_tensor,)


NODE_CLASS_MAPPINGS = {
    "DFloat11QwenImageLoader": DFloat11QwenImageLoader,
    "QwenImageTextEncode": QwenImageTextEncode,
    "QwenImageSampler": QwenImageSampler,
    "QwenImageAspectRatio": QwenImageAspectRatio,
    "QwenImagePresetSampler": QwenImagePresetSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DFloat11QwenImageLoader": "DFloat11 Qwen-Image Loader",
    "QwenImageTextEncode": "Qwen-Image Text Encode",
    "QwenImageSampler": "Qwen-Image Sampler",
    "QwenImageAspectRatio": "Qwen-Image Aspect Ratio",
    "QwenImagePresetSampler": "Qwen-Image Preset (All-in-One)",
}
