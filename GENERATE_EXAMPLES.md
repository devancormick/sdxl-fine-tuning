# Generate Client Examples

## Quick Start

To generate example images for client showcase:

```bash
# Activate virtual environment
source venv/bin/activate

# Generate examples (will auto-download models if needed)
python scripts/generate_client_examples.py \
    --max-examples 8 \
    --output outputs/client_examples \
    --fast-mode \
    --device cpu  # or "cuda" if GPU available
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
- `--use-pose`: Use pose images from `data/poses/` if available
- `--config PATH`: Custom config file

## Configuration

The script uses `config/inference_config.yaml` which has been updated to use HuggingFace model IDs for auto-download:

- Base model: `stabilityai/stable-diffusion-xl-base-1.0`
- ControlNet: `thibaud/controlnet-openpose-sdxl-1.0`
- VAE: `madebyollin/sdxl-vae-fp16-fix`

## Notes

- **First run**: Will download ~10GB of models (one-time, cached for future use)
- **CPU mode**: Much slower (30-60+ seconds per image), use GPU if available
- **GPU mode**: Fast generation (5-8 seconds per image with fast-mode)
- **No pose images**: Script creates placeholder images automatically

## Output

All generated images are saved to the output directory with descriptive names:
- `professional_portrait_01.png`
- `elegant_fashion_01.png`
- `character_studio_01.png`
- etc.

These can be used directly for client presentations and demonstrations.
