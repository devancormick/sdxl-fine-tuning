# SDXL Fine-Tuning Project

Production-ready SDXL fine-tuning pipeline with ControlNet pose control and video generation capabilities.

## Overview

This project provides a complete solution for fine-tuning SDXL-based models (like Endgame or Gonzalomo) with:
- **Fine-tuning** using LoRA for efficient training
- **ControlNet integration** for precise pose control
- **Multi-input support**: poses, attire, character, and background images
- **Fast inference**: 5-8 second image generation (1024x1024)
- **Video generation** from generated images

## Features

- ✅ LoRA-based fine-tuning for efficient training
- ✅ ControlNet pose control
- ✅ Multi-image input support (poses, attire, character, background)
- ✅ Optimized inference pipeline for production (5-8s generation time)
- ✅ Video generation from image sequences
- ✅ 1024x1024 resolution support
- ✅ Scalable pose dataset (supports 150+ poses, easily expandable)
- ✅ Batch processing API endpoint
- ✅ Training data validation and preparation tools
- ✅ Comprehensive logging system
- ✅ Docker containerization support
- ✅ Test suite with pytest

## Project Structure

```
sdxl-fine-tuning/
├── config/              # Configuration files
├── data/               # Training data
│   ├── poses/         # Pose reference images
│   ├── attire/        # Attire reference images
│   ├── characters/    # Character reference images
│   └── backgrounds/   # Background reference images
├── models/            # Saved models and checkpoints
├── outputs/           # Generated outputs
│   ├── images/       # Generated images
│   └── videos/       # Generated videos
├── scripts/           # Training and inference scripts
└── src/              # Source code
```

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- 50GB+ free disk space for models and data

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download base models:
```bash
python scripts/download_models.py
```

3. Prepare your training data:
   - Place pose images in `data/poses/`
   - Place attire images in `data/attire/`
   - Place character images in `data/characters/`
   - Place background images in `data/backgrounds/`

## Usage

### Training

1. Configure training parameters in `config/training_config.yaml`

2. Run fine-tuning:
```bash
python scripts/train_lora.py
```

### Inference

Generate images with pose control:

```bash
python scripts/generate_images.py \
    --pose data/poses/pose_001.png \
    --attire data/attire/attire_001.png \
    --character data/characters/char_001.png \
    --background data/backgrounds/bg_001.png \
    --prompt "a professional portrait" \
    --output outputs/images/
```

### Video Generation

Generate videos from image sequences:

```bash
python scripts/generate_video.py \
    --input outputs/images/ \
    --output outputs/videos/output.mp4 \
    --fps 24
```

## Configuration

Key configuration files:
- `config/training_config.yaml` - Training hyperparameters
- `config/inference_config.yaml` - Inference settings
- `config/model_config.yaml` - Model architecture settings

## Production Deployment

The inference pipeline is optimized for production use:
- Batch processing support
- GPU memory optimization
- Fast generation times (5-8 seconds)
- REST API option (see `src/api/`)

## License

See LICENSE file for details.

