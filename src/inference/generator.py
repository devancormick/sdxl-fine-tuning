"""Optimized inference pipeline for SDXL with ControlNet."""

import torch
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict
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
    extract_pose_keypoints,
)
from utils.model_utils import (
    select_base_model,
    verify_controlnet_model,
    verify_vae_model,
    load_model_config,
)


class SDXLImageGenerator:
    """Optimized SDXL image generator with ControlNet and LoRA support."""
    
    def __init__(
        self,
        base_model_path: Optional[str] = None,
        controlnet_model_path: Optional[str] = None,
        lora_weights_path: Optional[str] = None,
        vae_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        enable_optimizations: bool = True,
        extract_pose: bool = True,
        model_config_path: Optional[str] = None,
        preferred_model: Optional[str] = None,
        use_ip_adapter: bool = False,
        ip_adapter_scale: float = 0.8,
    ):
        """
        Initialize SDXL image generator.
        
        Args:
            base_model_path: Path to base SDXL model (overrides config if provided)
            controlnet_model_path: Path to ControlNet model (overrides config if provided)
            lora_weights_path: Path to LoRA weights
            vae_path: Path to VAE model (overrides config if provided)
            device: Device to run on ("cuda" or "cpu")
            dtype: Data type (torch.float16 or torch.float32)
            enable_optimizations: Enable performance optimizations
            extract_pose: Extract pose keypoints before generation
            model_config_path: Path to model config YAML file
            preferred_model: Preferred model name ("endgame", "gonzalomo", "sdxl_base")
            use_ip_adapter: Enable IP-Adapter for better character/attire conditioning
            ip_adapter_scale: IP-Adapter conditioning scale (0.0-1.0)
        """
        self.device = device
        self.dtype = dtype
        self.use_ip_adapter = use_ip_adapter
        self.ip_adapter_scale = ip_adapter_scale
        self.extract_pose = extract_pose
        
        # Load model configuration if provided
        model_config = None
        if model_config_path and Path(model_config_path).exists():
            model_config = load_model_config(model_config_path)
        
        # Select base model with verification and fallback
        if base_model_path:
            # Use provided path directly
            actual_base_model = base_model_path
            model_name = "custom"
            print(f"Using provided base model: {actual_base_model}")
        elif model_config and preferred_model:
            # Use model selection utility
            actual_base_model, model_name = select_base_model(
                model_config,
                preferred_model=preferred_model,
                fallback_to_default=True
            )
        else:
            # Use default
            actual_base_model = "stabilityai/stable-diffusion-xl-base-1.0"
            model_name = "sdxl_base"
            print(f"Using default SDXL base model: {actual_base_model}")
        
        # Select ControlNet model
        if controlnet_model_path:
            actual_controlnet = controlnet_model_path
        elif model_config:
            actual_controlnet = model_config.get("controlnet_models", {}).get(
                "openpose", "thibaud/controlnet-openpose-sdxl-1.0"
            )
        else:
            actual_controlnet = "thibaud/controlnet-openpose-sdxl-1.0"
        
        # Verify ControlNet model
        is_available, error = verify_controlnet_model(actual_controlnet)
        if not is_available:
            print(f"⚠ {error}")
            raise ValueError(f"ControlNet model not available: {actual_controlnet}")
        
        print(f"Loading ControlNet: {actual_controlnet}...")
        controlnet = ControlNetModel.from_pretrained(
            actual_controlnet,
            torch_dtype=dtype,
        )
        
        # Select VAE model
        if vae_path:
            actual_vae = vae_path
        elif model_config:
            actual_vae = model_config.get("vae_models", {}).get(
                "default", "madebyollin/sdxl-vae-fp16-fix"
            )
        else:
            actual_vae = "madebyollin/sdxl-vae-fp16-fix"
        
        # Verify VAE model (optional, may fail silently)
        is_available, error = verify_vae_model(actual_vae)
        if not is_available:
            print(f"⚠ {error}, using default VAE")
            actual_vae = "madebyollin/sdxl-vae-fp16-fix"
        
        print(f"Loading VAE: {actual_vae}...")
        vae = AutoencoderKL.from_pretrained(
            actual_vae,
            torch_dtype=dtype,
        )
        
        print(f"Loading SDXL pipeline with base model: {actual_base_model}...")
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            actual_base_model,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        )
        
        # Note: IP-Adapter for SDXL ControlNet is experimental
        # Current implementation uses enhanced prompt engineering for character/attire
        # Future enhancement: Integrate IP-Adapter Plus for SDXL when stable
        if use_ip_adapter:
            print("ℹ IP-Adapter for SDXL with ControlNet is experimental")
            print("ℹ Using enhanced prompt engineering with detailed descriptions")
            # Keep flag for future implementation but use prompt enhancement for now
            self.use_enhanced_prompts = True
        else:
            self.use_enhanced_prompts = True  # Always use enhanced prompts by default
        
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
        num_inference_steps: int = 20,  # Reduced default for faster generation
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        seed: Optional[int] = None,
        fast_mode: bool = False,
    ) -> Tuple[Image.Image, float]:
        """
        Generate image from prompt and pose.
        
        Args:
            fast_mode: If True, uses optimized settings for 5-8s generation (15 steps)
        
        Returns:
            Generated image and generation time in seconds
        """
        # Apply fast mode optimizations
        if fast_mode:
            num_inference_steps = 15
            guidance_scale = 7.0
        
        start_time = time.time()
        
        # Load pose image if path provided
        if isinstance(pose_image, str):
            pose_image = load_image(pose_image)
        
        # Extract pose keypoints if image provided
        if pose_image:
            # Extract OpenPose keypoints for better ControlNet conditioning
            if self.extract_pose:
                pose_image = extract_pose_keypoints(pose_image, use_openpose=True)
            # Resize pose image to target resolution
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
            reference_weight=reference_weight,
            use_detailed_descriptions=self.use_enhanced_prompts,
        )
        
        # Use pose for ControlNet (primary control)
        return self.generate(
            prompt=enhanced_prompt,
            pose_image=pose_image,
            **kwargs
        )

