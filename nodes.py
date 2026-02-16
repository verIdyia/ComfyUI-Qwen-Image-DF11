import torch
import numpy as np
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from transformers.modeling_utils import no_init_weights
from dfloat11 import DFloat11Model
import folder_paths
import os
from PIL import Image


class DFloat11QwenImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        # Get available Qwen models from the diffusion_models folder
        qwen_models = []
        diffusion_models_path = folder_paths.get_folder_paths("diffusion_models")
        
        for path in diffusion_models_path:
            if os.path.exists(path):
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path) and "qwen" in item.lower():
                        qwen_models.append(item)
        
        if not qwen_models:
            qwen_models = ["Place Qwen models in ComfyUI/models/diffusion_models/"]
        
        return {
            "required": {
                "model_path": (qwen_models, {"default": qwen_models[0] if qwen_models else ""}),
                "cpu_offload": ("BOOLEAN", {"default": False}),
                "pin_memory": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("QWEN_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Qwen-Image"

    def load_model(self, model_path, cpu_offload, pin_memory):
        if "Place Qwen models" in model_path:
            raise ValueError("Please download Qwen-Image model to ComfyUI/models/diffusion_models/ folder")
        
        # Find the full path to the model
        diffusion_models_path = folder_paths.get_folder_paths("diffusion_models")
        full_model_path = None
        
        for path in diffusion_models_path:
            candidate_path = os.path.join(path, model_path)
            if os.path.exists(candidate_path):
                full_model_path = candidate_path
                break
        
        if not full_model_path:
            raise ValueError(f"Model not found: {model_path}")
        
        # Check for required components
        required_components = ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
        missing_components = []
        
        for component in required_components:
            component_path = os.path.join(full_model_path, component)
            if not os.path.exists(component_path):
                missing_components.append(component)
        
        if missing_components:
            raise ValueError(f"Missing required components in {full_model_path}: {missing_components}. Please download the complete Qwen-Image model structure.")
        
        # Check if model_index.json exists
        model_index_path = os.path.join(full_model_path, "model_index.json")
        if not os.path.exists(model_index_path):
            raise ValueError(f"model_index.json not found in {full_model_path}. Please download the complete model.")
        
        # Load transformer config and create model
        with no_init_weights():
            transformer = QwenImageTransformer2DModel.from_config(
                QwenImageTransformer2DModel.load_config(
                    full_model_path, subfolder="transformer",
                ),
            ).to(torch.bfloat16)
        
        # Check if we have DFloat11 compressed transformer
        df11_transformer_path = os.path.join(full_model_path, "transformer", "diffusion_pytorch_model.safetensors")
        if os.path.exists(df11_transformer_path):
            # Load DFloat11 compressed transformer weights
            DFloat11Model.from_pretrained(
                os.path.join(full_model_path, "transformer"),
                device="cpu",
                cpu_offload=cpu_offload,
                pin_memory=pin_memory,
                bfloat16_model=transformer,
            )
        else:
            raise ValueError(f"DFloat11 compressed transformer not found at: {df11_transformer_path}")
        
        # Load the complete diffusion pipeline from local path
        pipe = QwenImagePipeline.from_pretrained(
            full_model_path,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        
        if cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        
        return (pipe,)


class QwenImageTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "A beautiful landscape"}),
                "language": (["en", "zh"], {"default": "en"}),
                "add_magic_prompt": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "encode_text"
    CATEGORY = "Qwen-Image"

    def encode_text(self, text, language, add_magic_prompt):
        positive_magic = {
            "en": "Ultra HD, 4K, cinematic composition.",
            "zh": "超清，4K，电影级构图"
        }
        
        if add_magic_prompt:
            positive_prompt = text + " " + positive_magic[language]
        else:
            positive_prompt = text
            
        negative_prompt = " "
        
        return (positive_prompt, negative_prompt)


class QwenImageSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN_MODEL",),
                "positive": ("STRING",),
                "negative": ("STRING",),
                "width": ("INT", {"default": 1328, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1328, "min": 64, "max": 2048, "step": 8}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 30.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "Qwen-Image"

    def sample(self, model, positive, negative, width, height, steps, cfg, seed):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        image = model(
            prompt=positive,
            negative_prompt=negative,
            width=width,
            height=height,
            num_inference_steps=steps,
            true_cfg_scale=cfg,
            generator=generator
        ).images[0]
        
        # Convert PIL image to tensor for ComfyUI
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        
        return (image_tensor,)


class QwenImageDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "qwen_image"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode_image"
    CATEGORY = "Qwen-Image"
    OUTPUT_NODE = True

    def decode_image(self, image, filename_prefix):
        # Convert tensor back to PIL and save
        image_np = (image.squeeze(0).numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        
        output_dir = folder_paths.get_output_directory()
        filename = f"{filename_prefix}_{torch.randint(0, 1000000, (1,)).item():06d}.png"
        filepath = os.path.join(output_dir, filename)
        
        pil_image.save(filepath)
        
        return (image,)


class QwenImageAspectRatio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "16:9"}),
            }
        }
    
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_dimensions"
    CATEGORY = "Qwen-Image"

    def get_dimensions(self, aspect_ratio):
        aspect_ratios = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1140),
            "3:4": (1140, 1472),
        }
        
        width, height = aspect_ratios[aspect_ratio]
        return (width, height)


class QwenImagePresetSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN_MODEL",),
                "positive": ("STRING",),
                "preset": (["fast", "balanced", "quality"], {"default": "balanced"}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "16:9"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample_preset"
    CATEGORY = "Qwen-Image"

    def sample_preset(self, model, positive, preset, aspect_ratio, seed):
        presets = {
            "fast": {"steps": 20, "cfg": 3.0},
            "balanced": {"steps": 50, "cfg": 4.0},
            "quality": {"steps": 80, "cfg": 5.0}
        }
        
        aspect_ratios = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1140),
            "3:4": (1140, 1472),
        }
        
        preset_config = presets[preset]
        width, height = aspect_ratios[aspect_ratio]
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        image = model(
            prompt=positive,
            negative_prompt=" ",
            width=width,
            height=height,
            num_inference_steps=preset_config["steps"],
            true_cfg_scale=preset_config["cfg"],
            generator=generator
        ).images[0]
        
        # Convert PIL image to tensor for ComfyUI
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        
        return (image_tensor,)


NODE_CLASS_MAPPINGS = {
    "DFloat11QwenImageLoader": DFloat11QwenImageLoader,
    "QwenImageTextEncode": QwenImageTextEncode,
    "QwenImageSampler": QwenImageSampler,
    "QwenImageDecode": QwenImageDecode,
    "QwenImageAspectRatio": QwenImageAspectRatio,
    "QwenImagePresetSampler": QwenImagePresetSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DFloat11QwenImageLoader": "DFloat11 Qwen-Image Loader",
    "QwenImageTextEncode": "Qwen-Image Text Encode",
    "QwenImageSampler": "Qwen-Image Sampler",
    "QwenImageDecode": "Qwen-Image Decode",
    "QwenImageAspectRatio": "Qwen-Image Aspect Ratio",
    "QwenImagePresetSampler": "Qwen-Image Preset Sampler",
}
