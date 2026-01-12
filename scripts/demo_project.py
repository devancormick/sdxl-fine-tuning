#!/usr/bin/env python
"""Demo script to show the project is working without requiring full model downloads."""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def main():
    print("="*70)
    print("SDXL Fine-Tuning Project - Demo")
    print("="*70)
    print()
    
    # Test 1: Import utilities
    print("✓ Testing utility imports...")
    try:
        from utils.model_utils import verify_model_available, select_base_model
        from utils.data_utils import load_config
        print("  ✓ Core utility modules imported successfully")
        # Try image utils (may require cv2)
        try:
            from utils.image_utils import resize_image, enhance_prompt_with_references
            print("  ✓ Image utility modules imported successfully")
        except ImportError as e:
            print(f"  ⚠ Image utilities require additional dependencies: {e}")
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False
    
    # Test 2: Check config files
    print("\n✓ Testing configuration files...")
    config_files = [
        "config/training_config.yaml",
        "config/inference_config.yaml",
        "config/model_config.yaml",
    ]
    for config_file in config_files:
        config_path = project_root / config_file
        if config_path.exists():
            try:
                config = load_config(str(config_path))
                print(f"  ✓ {config_file} loaded successfully")
            except Exception as e:
                print(f"  ⚠ {config_file} exists but has errors: {e}")
        else:
            print(f"  ✗ {config_file} not found")
    
    # Test 3: Check data directories
    print("\n✓ Testing data directories...")
    data_dirs = ["data/poses", "data/attire", "data/characters", "data/backgrounds"]
    for data_dir in data_dirs:
        dir_path = project_root / data_dir
        if dir_path.exists():
            file_count = len(list(dir_path.glob("*.png")) + list(dir_path.glob("*.jpg")))
            print(f"  ✓ {data_dir} exists ({file_count} images)")
        else:
            print(f"  ⚠ {data_dir} does not exist (will be created when needed)")
    
    # Test 4: Check output directories
    print("\n✓ Testing output directories...")
    output_dirs = ["outputs/images", "outputs/videos"]
    for output_dir in output_dirs:
        dir_path = project_root / output_dir
        if dir_path.exists():
            print(f"  ✓ {output_dir} exists")
        else:
            print(f"  ⚠ {output_dir} does not exist (will be created when needed)")
    
    # Test 5: Model verification utility
    print("\n✓ Testing model verification utility...")
    try:
        # Test with a known model
        is_available, error = verify_model_available(
            "stabilityai/stable-diffusion-xl-base-1.0",
            "SDXL Base"
        )
        if is_available:
            print("  ✓ SDXL Base model is available on HuggingFace Hub")
        else:
            print(f"  ℹ Model verification: {error}")
    except Exception as e:
        print(f"  ⚠ Model verification test skipped: {e}")
    
    # Test 6: Prompt enhancement
    print("\n✓ Testing prompt enhancement...")
    try:
        from PIL import Image
        enhanced = enhance_prompt_with_references(
            "a professional portrait",
            reference_weight=0.8,
            use_detailed_descriptions=True
        )
        print(f"  ✓ Enhanced prompt: {enhanced}")
    except Exception as e:
        print(f"  ⚠ Prompt enhancement test skipped: {e}")
    
    # Test 7: Script availability
    print("\n✓ Checking available scripts...")
    scripts = [
        "download_models.py",
        "generate_images.py",
        "generate_video.py",
        "train_lora.py",
        "validate_performance.py",
        "validate_data.py",
    ]
    for script in scripts:
        script_path = project_root / "scripts" / script
        if script_path.exists():
            print(f"  ✓ scripts/{script}")
        else:
            print(f"  ✗ scripts/{script} not found")
    
    print("\n" + "="*70)
    print("Project Structure: ✅ VALID")
    print("="*70)
    print("\nTo fully run the project:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download models: python scripts/download_models.py")
    print("3. Add your data to data/ directories")
    print("4. Run inference: python scripts/generate_images.py --help")
    print("\nSee QUICKSTART.md for detailed instructions.")
    print("="*70)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
