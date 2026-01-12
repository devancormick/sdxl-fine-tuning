"""Example: Batch image generation from pose directory."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from inference.generator import SDXLImageGenerator
from tqdm import tqdm
import torch


def main():
    # Initialize generator (load once)
    print("Loading generator...")
    generator = SDXLImageGenerator(
        base_model_path="stabilityai/stable-diffusion-xl-base-1.0",
        controlnet_model_path="thibaud/controlnet-openpose-sdxl-1.0",
        device="cuda",
        dtype=torch.float16,
        enable_optimizations=True,
    )
    
    # Settings
    pose_dir = Path("data/poses")
    output_dir = Path("outputs/images/batch")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prompt = "a professional portrait, high quality, detailed"
    negative_prompt = "blurry, low quality, distorted, bad anatomy"
    
    # Process all poses
    pose_images = sorted(list(pose_dir.glob("*.png")) + list(pose_dir.glob("*.jpg")))
    
    print(f"Generating {len(pose_images)} images...")
    
    total_time = 0
    for i, pose_path in enumerate(tqdm(pose_images)):
        try:
            image, gen_time = generator.generate(
                prompt=prompt,
                pose_image=str(pose_path),
                negative_prompt=negative_prompt,
                width=1024,
                height=1024,
                num_inference_steps=25,  # Reduced for speed
                guidance_scale=7.5,
                seed=42 + i,  # Vary seed for diversity
            )
            
            # Save image
            output_path = output_dir / f"{pose_path.stem}_generated.png"
            image.save(output_path)
            
            total_time += gen_time
            
        except Exception as e:
            print(f"\nError processing {pose_path}: {e}")
            continue
    
    avg_time = total_time / len(pose_images) if pose_images else 0
    print(f"\nCompleted! Generated {len(pose_images)} images")
    print(f"Average generation time: {avg_time:.2f}s")
    print(f"Average FPS: {1/avg_time:.2f}" if avg_time > 0 else "")


if __name__ == "__main__":
    main()

