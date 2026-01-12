#!/usr/bin/env python3
"""Generate example images for client showcase.

This script generates a variety of example images with different prompts
to demonstrate the capabilities of the SDXL fine-tuning project.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict
import time

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    import torch
    from PIL import Image
    from inference.generator import SDXLImageGenerator
    from utils.data_utils import load_config
except ImportError as e:
    print(f"Error: Required dependencies not installed. Please run: pip install -r requirements.txt")
    print(f"Missing module: {e}")
    sys.exit(1)


# Example prompts for client showcase
EXAMPLE_PROMPTS = [
    {
        "name": "professional_portrait_01",
        "prompt": "a professional portrait of a person, high quality, detailed, modern studio lighting, clean background",
        "negative_prompt": "blurry, low quality, distorted, bad anatomy, extra limbs",
        "description": "Professional portrait with clean background"
    },
    {
        "name": "elegant_fashion_01",
        "prompt": "an elegant fashion portrait, sophisticated style, high-end fashion photography, detailed clothing, professional lighting",
        "negative_prompt": "blurry, low quality, distorted, bad anatomy, amateur",
        "description": "Elegant fashion photography style"
    },
    {
        "name": "character_studio_01",
        "prompt": "a detailed character portrait, studio quality, professional photography, cinematic lighting, high resolution",
        "negative_prompt": "blurry, low quality, distorted, bad anatomy, artifacts",
        "description": "Character studio portrait"
    },
    {
        "name": "creative_concept_01",
        "prompt": "a creative artistic portrait, imaginative style, unique composition, professional photography, high quality",
        "negative_prompt": "blurry, low quality, distorted, bad anatomy, boring",
        "description": "Creative artistic portrait"
    },
    {
        "name": "modern_business_01",
        "prompt": "a modern business portrait, professional attire, contemporary setting, clean and polished, high quality",
        "negative_prompt": "blurry, low quality, distorted, bad anatomy, unprofessional",
        "description": "Modern business portrait"
    },
    {
        "name": "lifestyle_casual_01",
        "prompt": "a casual lifestyle portrait, natural lighting, relaxed atmosphere, authentic, high quality",
        "negative_prompt": "blurry, low quality, distorted, bad anatomy, staged",
        "description": "Casual lifestyle portrait"
    },
    {
        "name": "dramatic_lighting_01",
        "prompt": "a dramatic portrait with striking lighting, high contrast, cinematic quality, professional photography",
        "negative_prompt": "blurry, low quality, distorted, bad anatomy, flat lighting",
        "description": "Dramatic lighting portrait"
    },
    {
        "name": "minimalist_clean_01",
        "prompt": "a minimalist portrait, clean composition, simple background, professional quality, high resolution",
        "negative_prompt": "blurry, low quality, distorted, bad anatomy, cluttered",
        "description": "Minimalist clean portrait"
    },
]


def generate_examples(
    output_dir: Path,
    config_path: str = None,
    use_pose: bool = False,
    pose_dir: Path = None,
    fast_mode: bool = True,
    device: str = "cuda",
    max_examples: int = None,
):
    """Generate example images for client showcase.
    
    Args:
        output_dir: Directory to save generated images
        config_path: Path to inference config file
        use_pose: Whether to use pose images if available
        pose_dir: Directory containing pose images
        fast_mode: Use fast mode for quicker generation (5-8s target)
        device: Device to use ("cuda" or "cpu")
        max_examples: Maximum number of examples to generate (None for all)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config if provided
    config = {}
    if config_path and Path(config_path).exists():
        config = load_config(config_path)
    
    model_config = config.get("model", {})
    gen_config = config.get("generation", {})
    
    print("=" * 70)
    print("SDXL Fine-Tuning - Client Example Generation")
    print("=" * 70)
    print()
    
    # Initialize generator
    print("Initializing image generator...")
    try:
        generator = SDXLImageGenerator(
            base_model_path=model_config.get("base_model"),
            controlnet_model_path=model_config.get("controlnet_model"),
            lora_weights_path=model_config.get("lora_weights"),
            vae_path=model_config.get("vae_model"),
            device=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            enable_optimizations=True,
        )
        print("✓ Generator initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing generator: {e}")
        print("\nPlease ensure:")
        print("1. Dependencies are installed: pip install -r requirements.txt")
        print("2. Models are downloaded: python scripts/download_models.py")
        return False
    
    # Get pose images if using poses
    pose_images = []
    if use_pose and pose_dir and pose_dir.exists():
        pose_images = sorted(list(pose_dir.glob("*.png")) + list(pose_dir.glob("*.jpg")))
        print(f"✓ Found {len(pose_images)} pose images")
    
    # Select prompts to generate
    prompts_to_generate = EXAMPLE_PROMPTS
    if max_examples:
        prompts_to_generate = prompts_to_generate[:max_examples]
    
    print(f"\nGenerating {len(prompts_to_generate)} example images...")
    print("-" * 70)
    
    total_time = 0
    successful = 0
    failed = 0
    
    for i, example in enumerate(prompts_to_generate, 1):
        print(f"\n[{i}/{len(prompts_to_generate)}] {example['name']}")
        print(f"Description: {example['description']}")
        print(f"Prompt: {example['prompt'][:60]}...")
        
        try:
            # Select pose image if available (cycle through poses)
            pose_image_path = None
            if use_pose and pose_images:
                pose_idx = (i - 1) % len(pose_images)
                pose_image_path = str(pose_images[pose_idx])
                print(f"Using pose: {pose_images[pose_idx].name}")
            else:
                # Create a simple placeholder image for ControlNet (required by the pipeline)
                # A simple colored image will work as a minimal conditioning input
                placeholder = Image.new('RGB', (gen_config.get("width", 1024), gen_config.get("height", 1024)), color=(128, 128, 128))
                placeholder_path = output_dir / f"_placeholder_{i}.png"
                placeholder.save(placeholder_path)
                pose_image_path = str(placeholder_path)
                print("Using placeholder image (no pose provided)")
            
            # Generate image
            start_time = time.time()
            image, gen_time = generator.generate(
                prompt=example["prompt"],
                pose_image=pose_image_path,
                negative_prompt=example["negative_prompt"],
                width=gen_config.get("width", 1024),
                height=gen_config.get("height", 1024),
                num_inference_steps=gen_config.get("num_inference_steps", 20),
                guidance_scale=gen_config.get("guidance_scale", 7.5),
                fast_mode=fast_mode,
                seed=42 + i,  # Vary seed for diversity
            )
            
            # Save image
            output_path = output_dir / f"{example['name']}.png"
            image.save(output_path)
            
            # Clean up placeholder if used
            if not use_pose or not pose_images:
                placeholder_path = output_dir / f"_placeholder_{i}.png"
                if placeholder_path.exists():
                    placeholder_path.unlink()
            
            total_time += gen_time
            successful += 1
            
            print(f"✓ Generated in {gen_time:.2f}s")
            print(f"✓ Saved to: {output_path}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
    
    # Summary
    print("\n" + "=" * 70)
    print("Generation Complete")
    print("=" * 70)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if successful > 0:
        avg_time = total_time / successful
        print(f"Average generation time: {avg_time:.2f}s")
        print(f"Total time: {total_time:.2f}s")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 70)
    
    return successful > 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate example images for client showcase"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/client_examples",
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/inference_config.yaml",
        help="Path to inference config file"
    )
    parser.add_argument(
        "--use-pose",
        action="store_true",
        help="Use pose images if available in data/poses/"
    )
    parser.add_argument(
        "--pose-dir",
        type=str,
        default="data/poses",
        help="Directory containing pose images"
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        default=True,
        help="Use fast mode (5-8s generation time)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for generation"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Maximum number of examples to generate"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    output_dir = Path(args.output)
    pose_dir = Path(args.pose_dir) if args.use_pose else None
    
    # Generate examples
    success = generate_examples(
        output_dir=output_dir,
        config_path=args.config,
        use_pose=args.use_pose,
        pose_dir=pose_dir,
        fast_mode=args.fast_mode,
        device=args.device,
        max_examples=args.max_examples,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
