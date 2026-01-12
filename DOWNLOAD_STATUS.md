# Model Download Status

## Models Being Downloaded

The download script will fetch the following models from HuggingFace:

1. **SDXL Base Model** (`stabilityai/stable-diffusion-xl-base-1.0`)
   - Size: ~6-7 GB
   - Status: ✅ Directory created, downloading model weights...

2. **ControlNet OpenPose** (`thibaud/controlnet-openpose-sdxl-1.0`)
   - Size: ~2-3 GB
   - Status: ⏳ Pending

3. **VAE Model** (`madebyollin/sdxl-vae-fp16-fix`)
   - Size: ~300 MB
   - Status: ⏳ Pending

**Total download size: ~10-12 GB**

## How to Download

### Option 1: Run the download script
```bash
cd sdxl-fine-tuning
python scripts/download_models.py
```

### Option 2: Download specific model
```bash
python scripts/download_models.py --model stabilityai/stable-diffusion-xl-base-1.0
```

### Option 3: Use HuggingFace CLI
```bash
# Install huggingface-cli if not already installed
pip install huggingface_hub[cli]

# Login to HuggingFace (optional, for private models)
huggingface-cli login

# Download models
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --local-dir models/base_models/stable-diffusion-xl-base-1.0
huggingface-cli download thibaud/controlnet-openpose-sdxl-1.0 --local-dir models/controlnet-openpose-sdxl-1.0
huggingface-cli download madebyollin/sdxl-vae-fp16-fix --local-dir models/vae/sdxl-vae-fp16-fix
```

## Verification

After download, verify models are present:

```bash
# Check base model
ls models/base_models/stable-diffusion-xl-base-1.0/unet/*.safetensors

# Check ControlNet
ls models/controlnet-openpose-sdxl-1.0/*.safetensors

# Check VAE
ls models/vae/sdxl-vae-fp16-fix/*.safetensors
```

## Troubleshooting

### Download is slow
- Models are large (10+ GB total)
- Speed depends on internet connection
- Downloads are resumable - if interrupted, restart and it will continue

### Out of disk space
- Ensure you have at least 15-20 GB free space
- Models are downloaded to `models/` directory

### Permission errors
- Ensure you have write permissions to `models/` directory
- On Windows, run as administrator if needed

### Network errors
- Check internet connection
- Some corporate networks may block large downloads
- Try using a VPN if needed

