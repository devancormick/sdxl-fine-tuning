# Generate Client Examples

## Quick Start

To generate example images for client showcase:

```bash
# Activate virtual environment
source venv/bin/activate

# Generate examples (will auto-download models if needed)
# Parallel processing is enabled by default (auto-detects CPU cores)
python scripts/generate_client_examples.py \
    --max-examples 8 \
    --output outputs/client_examples \
    --fast-mode \
    --device cpu  # or "cuda" if GPU available

# Or specify number of workers manually
python scripts/generate_client_examples.py \
    --max-examples 8 \
    --output outputs/client_examples \
    --fast-mode \
    --device cpu \
    --num-workers 4  # Use 4 parallel workers
```

## Script Features

The `generate_client_examples.py` script:

- ✅ Generates 8 different example images with various styles:
  - Professional portraits
  - Fashion photography
  - Character studio shots
  - Creative artistic portraits
  - Business portraits
  - Lifestyle portraits
  - Dramatic lighting
  - Minimalist clean style

- ✅ Auto-downloads models from HuggingFace (if not already cached)
- ✅ Works with or without pose images
- ✅ **Parallel processing** with multiple CPU workers for faster generation
- ✅ Uses fast mode for 5-8 second generation (with GPU)
- ✅ Saves all examples to specified output directory

## Options

```bash
python scripts/generate_client_examples.py --help
```

Key options:
- `--max-examples N`: Generate only N examples (default: all 8)
- `--output DIR`: Output directory (default: `outputs/client_examples`)
- `--fast-mode`: Use fast generation settings (15 steps)
- `--device cpu|cuda`: Device to use
- `--num-workers N`: Number of parallel workers (default: auto-detect based on CPU cores)
- `--use-pose`: Use pose images from `data/poses/` if available
- `--config PATH`: Custom config file

## Configuration

The script uses `config/inference_config.yaml` which has been updated to use HuggingFace model IDs for auto-download:

- Base model: `stabilityai/stable-diffusion-xl-base-1.0`
- ControlNet: `thibaud/controlnet-openpose-sdxl-1.0`
- VAE: `madebyollin/sdxl-vae-fp16-fix`

## Notes

- **First run**: Will download ~10GB of models (one-time, cached for future use)
- **Parallel processing**: Automatically uses multiple CPU cores (up to 4 workers on CPU)
  - CPU mode with 4 workers: ~10-15 minutes for 8 images (vs ~40-60 minutes sequential)
  - GPU mode: Uses up to 2 workers (memory limited)
- **CPU mode**: Much slower (30-60+ seconds per image per worker), but parallel processing speeds it up
- **GPU mode**: Fast generation (5-8 seconds per image with fast-mode)
- **No pose images**: Script creates placeholder images automatically
- **Memory usage**: Each worker loads its own model instance (uses more RAM but faster)

## Output

All generated images are saved to the output directory with descriptive names:
- `professional_portrait_01.png`
- `elegant_fashion_01.png`
- `character_studio_01.png`
- etc.

These can be used directly for client presentations and demonstrations.
