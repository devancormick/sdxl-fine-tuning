# Quick Start Guide

## Setup

1. **Install dependencies:**
```bash
cd sdxl-fine-tuning
pip install -r requirements.txt
```

2. **Download base models:**
```bash
python scripts/download_models.py
```

This will download:
- Stable Diffusion XL base model
- ControlNet OpenPose model
- VAE model

## Prepare Your Data

Organize your training data in the `data/` directory:

```
data/
├── poses/        # Place ~150 pose images here
├── attire/       # Attire reference images
├── characters/   # Character reference images
└── backgrounds/  # Background reference images
```

## Training (Optional)

If you want to fine-tune the model with your data:

```bash
python scripts/train_lora.py --config config/training_config.yaml --data_dir data
```

Training will create LoRA weights in `models/lora_checkpoints/`.

## Generate Images

### Basic generation with pose:

```bash
python scripts/generate_images.py \
    --prompt "a professional portrait, high quality" \
    --pose data/poses/pose_001.png \
    --output outputs/images/
```

### With all inputs:

```bash
python scripts/generate_images.py \
    --prompt "a professional portrait in a modern setting" \
    --pose data/poses/pose_001.png \
    --attire data/attire/attire_001.png \
    --character data/characters/char_001.png \
    --background data/backgrounds/bg_001.png \
    --output outputs/images/
```

### With custom settings:

```bash
python scripts/generate_images.py \
    --prompt "your prompt here" \
    --pose data/poses/pose_001.png \
    --steps 25 \
    --guidance-scale 8.0 \
    --seed 42 \
    --output outputs/images/
```

## Generate Videos

### From directory of images:

```bash
python scripts/generate_video.py \
    --input outputs/images/ \
    --output outputs/videos/output.mp4 \
    --fps 24
```

### With transitions:

```bash
python scripts/generate_video.py \
    --input outputs/images/ \
    --output outputs/videos/output.mp4 \
    --fps 24 \
    --transition-frames 5 \
    --loop
```

### Slideshow mode:

```bash
python scripts/generate_video.py \
    --input outputs/images/ \
    --output outputs/videos/slideshow.mp4 \
    --fps 24 \
    --duration-per-image 3.0 \
    --transition-duration 0.5
```

## Performance Tips

1. **For faster generation (5-8s target):**
   - Use 20-25 inference steps (instead of 30)
   - Enable xformers (automatic if available)
   - Use fp16 precision
   - Reduce resolution if needed (1024x1024 recommended)

2. **If you run out of GPU memory:**
   - Enable VAE slicing (in `config/inference_config.yaml`)
   - Enable CPU offload
   - Reduce batch size

3. **For better quality:**
   - Use 30-50 inference steps
   - Fine-tune with your data
   - Use appropriate guidance scale (7-10)

## Troubleshooting

### Model download issues:
- Ensure you have enough disk space (50GB+)
- Check internet connection
- Models are downloaded to `~/.cache/huggingface/` by default

### CUDA out of memory:
- Reduce resolution
- Enable optimizations in config
- Use smaller models

### Slow generation:
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU utilization
- Reduce inference steps
- Enable torch.compile if PyTorch 2.0+

## Next Steps

- Fine-tune with your specific data
- Experiment with different ControlNet models (Canny, Depth)
- Add IP-Adapter for better character/attire conditioning
- Set up batch processing for production

