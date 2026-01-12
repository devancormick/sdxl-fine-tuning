"""Optimized inference pipeline for SDXL with ControlNet."""

import torch
from pathlib import Path
from typing import Optional, Union, List, Tuple
from PIL import Image
import time

from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
)
from diffusers.utils import load_image
from peft import PeftModel
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.image_utils import (
    preprocess_image_for_controlnet,
    resize_image,
    enhance_prompt_with_references,
)


class SDXLImageGenerator:
    """Optimized SDXL image generator with ControlNet and LoRA support."""
    
    def __init__(
        self,
        base_model_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet_model_path: str = "thibaud/controlnet-openpose-sdxl-1.0",
        lora_weights_path: Optional[str] = None,
        vae_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        enable_optimizations: bool = True,
    ):
        self.device = device
        self.dtype = dtype
        
        print("Loading ControlNet...")
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_path,
            torch_dtype=dtype,
        )
        
        print("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            vae_path or "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=dtype,
        )
        
        print("Loading SDXL pipeline...")
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model_path,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        )
        
        # Load LoRA weights if provided
        if lora_weights_path and Path(lora_weights_path).exists():
            print(f"Loading LoRA weights from {lora_weights_path}...")
            self.pipe.unet = PeftModel.from_pretrained(
                self.pipe.unet,
                lora_weights_path,
                torch_dtype=dtype,
            )
            self.pipe.unet = self.pipe.unet.merge_and_unload()
        
        # Move to device
        self.pipe = self.pipe.to(device)
        
        # Apply optimizations
        if enable_optimizations:
            self._apply_optimizations()
        
        # Use faster scheduler
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        print("Pipeline loaded and optimized!")
    
    def _apply_optimizations(self):
        """Apply performance optimizations."""
        # Enable attention slicing for memory efficiency
        self.pipe.enable_attention_slicing(slice_size="max")
        
        # Enable VAE slicing
        self.pipe.enable_vae_slicing()
        
        # Try to enable xformers if available
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"Could not enable xformers: {e}")
        
        # Compile model if PyTorch 2.0+
        if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
            try:
                self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead")
                print("Model compiled with torch.compile")
            except Exception as e:
                print(f"Could not compile model: {e}")
    
    def generate(
        self,
        prompt: str,
        pose_image: Optional[Union[str, Image.Image]] = None,
        negative_prompt: str = "blurry, low quality, distorted, bad anatomy",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> Tuple[Image.Image, float]:
        """
        Generate image from prompt and pose.
        
        Returns:
            Generated image and generation time in seconds
        """
        start_time = time.time()
        
        # Load pose image if path provided
        if isinstance(pose_image, str):
            pose_image = load_image(pose_image)
        
        # Resize pose image to target resolution
        if pose_image:
            pose_image = resize_image(pose_image, (width, height))
        
        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate image
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                image=pose_image,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
            )
        
        generation_time = time.time() - start_time
        
        return result.images[0], generation_time
    
    def generate_with_multi_inputs(
        self,
        prompt: str,
        pose_image: Optional[Union[str, Image.Image]] = None,
        attire_image: Optional[Union[str, Image.Image]] = None,
        character_image: Optional[Union[str, Image.Image]] = None,
        background_image: Optional[Union[str, Image.Image]] = None,
        reference_weight: float = 0.5,
        **kwargs
    ) -> Tuple[Image.Image, float]:
        """
        Generate image with multiple input references.
        
        Uses ControlNet for pose control and enhanced prompt engineering for
        character, attire, and background references.
        
        Args:
            prompt: Base text prompt
            pose_image: Pose reference image (used with ControlNet)
            attire_image: Attire reference image (used to enhance prompt)
            character_image: Character reference image (used to enhance prompt)
            background_image: Background reference image (used to enhance prompt)
            reference_weight: Weight for reference-based prompt enhancement (0.0-1.0)
            **kwargs: Additional arguments passed to generate()
        
        Returns:
            Generated image and generation time
        """
        # Load images if paths provided
        char_img = None
        attr_img = None
        bg_img = None
        
        if character_image:
            char_img = load_image(character_image) if isinstance(character_image, str) else character_image
        if attire_image:
            attr_img = load_image(attire_image) if isinstance(attire_image, str) else attire_image
        if background_image:
            bg_img = load_image(background_image) if isinstance(background_image, str) else background_image
        
        # Enhance prompt with reference descriptions
        enhanced_prompt = enhance_prompt_with_references(
            prompt,
            character_image=char_img,
            attire_image=attr_img,
            background_image=bg_img,
        )
        
        # Use pose for ControlNet (primary control)
        return self.generate(
            prompt=enhanced_prompt,
            pose_image=pose_image,
            **kwargs
        )

