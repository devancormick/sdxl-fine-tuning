#!/usr/bin/env python
"""Download required models for SDXL fine-tuning."""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))
from utils.data_utils import load_config


def download_model(model_id: str, local_dir: Path, revision: str = "main"):
    """Download a model from HuggingFace."""
    print(f"Downloading {model_id} to {local_dir}...")
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            revision=revision,
            local_dir_use_symlinks=False,
        )
        print(f"✓ Downloaded {model_id}")
    except Exception as e:
        print(f"✗ Error downloading {model_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download SDXL models")
    parser.add_argument("--config", type=str, default="config/model_config.yaml", help="Model config file")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for models")
    parser.add_argument("--model", type=str, help="Specific model to download (optional)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = load_config(args.config) if Path(args.config).exists() else {}
    
    models_to_download = []
    
    if args.model:
        # Download specific model
        models_to_download.append((args.model, output_dir / args.model.replace("/", "_")))
    else:
        # Download all models from config
        base_models = config.get("base_models", {})
        controlnet_models = config.get("controlnet_models", {})
        vae_models = config.get("vae_models", {})
        
        # Base model (required)
        models_to_download.append((
            base_models.get("sdxl_base", "stabilityai/stable-diffusion-xl-base-1.0"),
            output_dir / "base_models" / "stable-diffusion-xl-base-1.0"
        ))
        
        # ControlNet (required)
        models_to_download.append((
            controlnet_models.get("openpose", "thibaud/controlnet-openpose-sdxl-1.0"),
            output_dir / "controlnet-openpose-sdxl-1.0"
        ))
        
        # VAE (required)
        models_to_download.append((
            vae_models.get("default", "madebyollin/sdxl-vae-fp16-fix"),
            output_dir / "vae" / "sdxl-vae-fp16-fix"
        ))
    
    print(f"Downloading {len(models_to_download)} model(s)...\n")
    
    for model_id, local_dir in models_to_download:
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        download_model(model_id, local_dir)
        print()
    
    print("Download complete!")


if __name__ == "__main__":
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed. Install it with:")
        print("  pip install huggingface_hub")
        sys.exit(1)
    
    main()

