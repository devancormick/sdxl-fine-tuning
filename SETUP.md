# Setup Instructions

Complete setup guide for the SDXL Fine-Tuning project.

## Prerequisites

### System Requirements
- **OS**: Linux (recommended), Windows, or macOS
- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with CUDA support (12GB+ VRAM recommended)
- **RAM**: 16GB+ (32GB+ recommended)
- **Disk Space**: 100GB+ free space

### Software Dependencies
- **CUDA**: 11.8 or 12.1+ (for GPU acceleration)
- **cuDNN**: Compatible with your CUDA version
- **Git**: For cloning/downloading the project

## Step 1: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Optional: Install API Dependencies

If you want to run the API server:

```bash
pip install fastapi uvicorn python-multipart pydantic
```

Or uncomment the API dependencies in `requirements.txt`.

## Step 2: Verify CUDA Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

If CUDA is not available, you can still run on CPU (much slower).

## Step 3: Download Models

Download the required models:

```bash
python scripts/download_models.py
```

This will download:
- Stable Diffusion XL base model (~6-7 GB)
- ControlNet OpenPose model (~2-3 GB)
- VAE model (~300 MB)

**Total download size: ~10-12 GB**

Models are downloaded to `~/.cache/huggingface/hub/` by default.

### Manual Model Download

If the script doesn't work, you can download models manually using:

```python
from huggingface_hub import snapshot_download

# Base model
snapshot_download(repo_id="stabilityai/stable-diffusion-xl-base-1.0", local_dir="models/base_models/sdxl")

# ControlNet
snapshot_download(repo_id="thibaud/controlnet-openpose-sdxl-1.0", local_dir="models/controlnet")

# VAE
snapshot_download(repo_id="madebyollin/sdxl-vae-fp16-fix", local_dir="models/vae")
```

## Step 4: Prepare Training Data

Organize your training data:

```
data/
├── poses/        # Place your ~150 pose images here (.png or .jpg)
├── attire/       # Attire reference images
├── characters/   # Character reference images
└── backgrounds/  # Background reference images
```

### Data Requirements

- **Format**: PNG or JPG
- **Recommended resolution**: 1024x1024 or higher
- **Poses**: ~150 images (can add more later)
- **Other categories**: As many as needed

### Prepare Captions (Optional)

If you want to use custom captions:

```python
from src.utils.data_utils import prepare_captions

prepare_captions("data", "captions.json")
```

## Step 5: Configure Settings

Review and update configuration files:

1. **Training config**: `config/training_config.yaml`
   - Adjust learning rate, batch size, etc.
   - Set output directory

2. **Inference config**: `config/inference_config.yaml`
   - Adjust generation parameters
   - Set model paths
   - Enable/disable optimizations

3. **Model config**: `config/model_config.yaml`
   - Change base model if needed
   - Select ControlNet variant

## Step 6: Test Installation

### Test image generation:

```bash
python scripts/generate_images.py \
    --prompt "a beautiful landscape" \
    --output outputs/test/
```

If you have a pose image:

```bash
python scripts/generate_images.py \
    --prompt "a professional portrait" \
    --pose data/poses/your_pose.png \
    --output outputs/test/
```

### Test video generation:

```bash
# First generate some test images, then:
python scripts/generate_video.py \
    --input outputs/test/ \
    --output outputs/test_video.mp4 \
    --fps 24
```

## Step 7: (Optional) Fine-tune Model

If you have training data:

```bash
python scripts/train_lora.py \
    --config config/training_config.yaml \
    --data_dir data
```

Training will create LoRA weights in `models/lora_checkpoints/`.

**Note**: Training requires significant GPU memory (16GB+ recommended). Training time depends on dataset size and hardware.

## Troubleshooting

### CUDA Out of Memory

- Reduce batch size in training config
- Enable gradient checkpointing
- Use smaller resolution (512x512 for testing)
- Enable CPU offload in inference config

### Model Download Fails

- Check internet connection
- Ensure sufficient disk space
- Try downloading models manually
- Check HuggingFace Hub access

### Import Errors

- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`
- Check Python version: `python --version` (should be 3.10+)

### Slow Generation

- Ensure GPU is being used (check `nvidia-smi`)
- Enable optimizations in config
- Reduce inference steps (20-25 instead of 30)
- Check GPU drivers are up to date

### Permission Errors

- On Linux/Mac: Use `chmod +x scripts/*.py` to make scripts executable
- Check write permissions for `models/` and `outputs/` directories

## Next Steps

1. Read `QUICKSTART.md` for usage examples
2. Read `DEPLOYMENT.md` for production deployment
3. Review `README.md` for project overview
4. Start fine-tuning with your data!

## Getting Help

- Check existing documentation
- Review configuration files
- Check GitHub issues (if project is on GitHub)
- Review HuggingFace documentation for models

