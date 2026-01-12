# Model Selection and Configuration Guide

## Overview

This project supports multiple SDXL-based models including:
- **SDXL Base** (default): Stability AI's Stable Diffusion XL
- **Endgame**: Custom SDXL model (if available)
- **Gonzalomo**: Custom SDXL model (if available)

The system automatically verifies model availability and falls back to SDXL Base if your preferred model is not available.

## Configuration

### Model Config File (`config/model_config.yaml`)

```yaml
base_models:
  sdxl_base: "stabilityai/stable-diffusion-xl-base-1.0"
  sdxl_refiner: "stabilityai/stable-diffusion-xl-refiner-1.0"
  gonzalomo: "gonzalomo/gonzalomo-xl"  # If available
  endgame: "endgame/SDXL-model"  # If available
```

### Using Model Selection

#### Via Command Line

```bash
# Use Endgame model
python scripts/generate_images.py \
    --prompt "your prompt" \
    --preferred-model endgame \
    --pose data/poses/pose_001.png

# Use Gonzalomo model
python scripts/generate_images.py \
    --prompt "your prompt" \
    --preferred-model gonzalomo \
    --pose data/poses/pose_001.png

# Use SDXL Base (default)
python scripts/generate_images.py \
    --prompt "your prompt" \
    --preferred-model sdxl_base \
    --pose data/poses/pose_001.png
```

#### Via Code

```python
from src.inference.generator import SDXLImageGenerator

# Initialize with preferred model
generator = SDXLImageGenerator(
    model_config_path="config/model_config.yaml",
    preferred_model="endgame",  # or "gonzalomo", "sdxl_base"
    device="cuda",
)
```

## Model Verification

The system automatically verifies model availability:

1. **HuggingFace Models**: Checks if the model exists on HuggingFace Hub
2. **Local Models**: Checks if the model exists as a local path
3. **Fallback**: Automatically falls back to SDXL Base if preferred model unavailable

### Verification Output

```
✓ Using endgame model: endgame/SDXL-model
Loading ControlNet: thibaud/controlnet-openpose-sdxl-1.0...
Loading VAE: madebyollin/sdxl-vae-fp16-fix...
```

If model is not available:
```
⚠ Base model (endgame) 'endgame/SDXL-model' not found: ...
⚠ Falling back to default SDXL base model: stabilityai/stable-diffusion-xl-base-1.0
```

## Adding Custom Models

To add a custom model:

1. **Add to config** (`config/model_config.yaml`):
```yaml
base_models:
  my_custom_model: "username/my-custom-sdxl-model"
```

2. **Use in code**:
```python
generator = SDXLImageGenerator(
    preferred_model="my_custom_model",
    model_config_path="config/model_config.yaml",
)
```

## ControlNet Training Approach

### Current Implementation

The project uses a **two-stage approach**:

1. **Stage 1: LoRA Training**
   - Fine-tune LoRA adapters on the base SDXL model
   - Trains on your specific data (poses, characters, attire, backgrounds)
   - Preserves base model quality while adding custom style/character

2. **Stage 2: ControlNet Integration**
   - ControlNet is applied during inference (not during training)
   - Provides precise pose control
   - Works with any LoRA-trained model

### Why This Approach?

- **Efficiency**: LoRA training is much faster than full model training
- **Flexibility**: Can use different ControlNet models (OpenPose, Canny, Depth)
- **Quality**: Maintains SDXL's high quality while adding control
- **Compatibility**: Works with any SDXL-compatible base model

### Training Workflow

```bash
# 1. Prepare your data in data/ directory
data/
├── poses/
├── attire/
├── characters/
└── backgrounds/

# 2. Train LoRA
python scripts/train_lora.py \
    --config config/training_config.yaml \
    --data_dir data

# 3. Generate with LoRA + ControlNet
python scripts/generate_images.py \
    --prompt "your prompt" \
    --pose data/poses/pose_001.png \
    --lora-weights models/lora_checkpoints/checkpoint-2000 \
    --character data/characters/char_001.png
```

## Performance Considerations

- **Model Loading**: First generation is slower due to model loading
- **Generation Speed**: 5-8 seconds per image with fast mode (15 steps)
- **Memory Usage**: ~12-16GB VRAM recommended for SDXL + ControlNet + LoRA

## Troubleshooting

### Model Not Found

If you get "model not found" errors:

1. **Check HuggingFace**: Verify the model ID exists on huggingface.co
2. **Check Local**: If using local path, verify it exists
3. **Check Config**: Ensure model name matches config file
4. **Use Fallback**: System will automatically fallback to SDXL Base

### Model Loading Errors

```python
# Verify model manually
from src.utils.model_utils import verify_model_available

is_available, error = verify_model_available("endgame/SDXL-model")
print(f"Available: {is_available}, Error: {error}")
```

### Custom Model Integration

For custom models not on HuggingFace:

1. **Local Path**: Use absolute path in config
2. **HuggingFace Upload**: Upload your model to HuggingFace Hub
3. **Private Repos**: Use HuggingFace tokens for private models

## Best Practices

1. **Always use fallback**: Set `fallback_to_default=True` (default)
2. **Verify models first**: Check availability before training
3. **Test generation**: Validate model works before production
4. **Document custom models**: Note any special requirements
5. **Version control**: Track which models you've tested

## Future Enhancements

- IP-Adapter integration for better character/attire conditioning
- Direct ControlNet training (experimental)
- Multi-model ensemble support
- Model performance benchmarking
