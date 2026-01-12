#!/usr/bin/env python
"""Image generation script with ControlNet and LoRA."""

import argparse
import sys
from pathlib import Path
from PIL import Image
import torch

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from inference.generator import SDXLImageGenerator
from utils.data_utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Generate images with SDXL + ControlNet + LoRA")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--pose", type=str, help="Path to pose reference image")
    parser.add_argument("--attire", type=str, help="Path to attire reference image")
    parser.add_argument("--character", type=str, help="Path to character reference image")
    parser.add_argument("--background", type=str, help="Path to background reference image")
    parser.add_argument("--negative-prompt", type=str, default="blurry, low quality, distorted, bad anatomy")
    parser.add_argument("--output", type=str, default="outputs/images/", help="Output directory")
    parser.add_argument("--config", type=str, default="config/inference_config.yaml", help="Config file")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--base-model", type=str, help="Override base model path")
    parser.add_argument("--lora-weights", type=str, help="Override LoRA weights path")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config) if Path(args.config).exists() else {}
    model_config = config.get("model", {})
    gen_config = config.get("generation", {})
    
    # Initialize generator
    print("Initializing image generator...")
    generator = SDXLImageGenerator(
        base_model_path=args.base_model or model_config.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0"),
        controlnet_model_path=model_config.get("controlnet_model", "thibaud/controlnet-openpose-sdxl-1.0"),
        lora_weights_path=args.lora_weights or model_config.get("lora_weights"),
        vae_path=model_config.get("vae_model"),
        device=config.get("device", "cuda"),
        dtype=torch.float16,
        enable_optimizations=True,
    )
    
    # Generate image
    print(f"Generating image with prompt: {args.prompt}")
    image, gen_time = generator.generate_with_multi_inputs(
        prompt=args.prompt,
        pose_image=args.pose,
        attire_image=args.attire,
        character_image=args.character,
        background_image=args.background,
        negative_prompt=args.negative_prompt or gen_config.get("negative_prompt", ""),
        width=args.width or gen_config.get("width", 1024),
        height=args.height or gen_config.get("height", 1024),
        num_inference_steps=args.steps or gen_config.get("num_inference_steps", 30),
        guidance_scale=args.guidance_scale or gen_config.get("guidance_scale", 7.5),
        controlnet_conditioning_scale=gen_config.get("controlnet_conditioning_scale", 1.0),
        seed=args.seed,
    )
    
    # Save image
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename from prompt
    filename = args.prompt.replace(" ", "_").replace(",", "")[:50] + ".png"
    output_path = output_dir / filename
    
    image.save(output_path)
    print(f"\nImage saved to: {output_path}")
    print(f"Generation time: {gen_time:.2f} seconds")
    print(f"FPS: {1/gen_time:.2f}")


if __name__ == "__main__":
    main()

