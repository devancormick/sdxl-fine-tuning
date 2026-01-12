# Project Structure

Complete overview of the SDXL Fine-Tuning project structure.

## Directory Layout

```
sdxl-fine-tuning/
├── config/                      # Configuration files
│   ├── training_config.yaml    # Training hyperparameters
│   ├── inference_config.yaml   # Inference settings
│   └── model_config.yaml       # Model paths and settings
│
├── data/                       # Training data directory
│   ├── poses/                  # ~150 pose reference images
│   ├── attire/                 # Attire reference images
│   ├── characters/             # Character reference images
│   └── backgrounds/            # Background reference images
│
├── models/                     # Model storage (gitignored)
│   ├── base_models/            # Base SDXL models
│   ├── lora_checkpoints/       # Fine-tuned LoRA weights
│   ├── controlnet/             # ControlNet models
│   └── vae/                    # VAE models
│
├── outputs/                    # Generated outputs (gitignored)
│   ├── images/                 # Generated images
│   ├── videos/                 # Generated videos
│   └── api/                    # API-generated outputs
│
├── scripts/                    # Executable scripts
│   ├── train_lora.py          # LoRA fine-tuning script
│   ├── generate_images.py     # Image generation script
│   ├── generate_video.py      # Video generation script
│   ├── download_models.py     # Model download script
│   └── run_api.py             # API server script
│
├── src/                        # Source code
│   ├── api/                    # API server code
│   │   ├── __init__.py
│   │   └── server.py          # FastAPI server
│   │
│   ├── inference/              # Inference modules
│   │   ├── __init__.py
│   │   ├── generator.py       # SDXL image generator with ControlNet
│   │   └── video_generator.py # Video generation from images
│   │
│   ├── training/               # Training modules
│   │   ├── __init__.py
│   │   └── train_lora.py      # LoRA training implementation
│   │
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── data_utils.py      # Dataset and data loading utilities
│       └── image_utils.py     # Image processing utilities
│
├── examples/                   # Example scripts
│   ├── __init__.py
│   └── batch_generation.py    # Batch processing example
│
├── .gitignore                  # Git ignore rules
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
│
└── Documentation/
    ├── README.md              # Project overview and features
    ├── QUICKSTART.md          # Quick start guide
    ├── SETUP.md               # Detailed setup instructions
    ├── DEPLOYMENT.md          # Production deployment guide
    └── PROJECT_STRUCTURE.md   # This file
```

## Key Files

### Configuration Files

- **`config/training_config.yaml`**: Training hyperparameters, LoRA settings, optimizer configuration
- **`config/inference_config.yaml`**: Generation parameters, optimization settings, device configuration
- **`config/model_config.yaml`**: Model repository paths and versions

### Core Source Files

#### Training
- **`src/training/train_lora.py`**: LoRA fine-tuning implementation using PEFT and Diffusers
- **`scripts/train_lora.py`**: Entry point for training script

#### Inference
- **`src/inference/generator.py`**: Main SDXL image generator with ControlNet and LoRA support
- **`src/inference/video_generator.py`**: Video generation from image sequences
- **`scripts/generate_images.py`**: CLI for image generation
- **`scripts/generate_video.py`**: CLI for video generation

#### API
- **`src/api/server.py`**: FastAPI server for production deployment
- **`scripts/run_api.py`**: Entry point for API server

#### Utilities
- **`src/utils/data_utils.py`**: Dataset classes and data loading utilities
- **`src/utils/image_utils.py`**: Image preprocessing and conversion utilities

### Scripts

All scripts in `scripts/` are executable entry points:

- **`train_lora.py`**: Fine-tune SDXL with LoRA
- **`generate_images.py`**: Generate images with ControlNet and LoRA
- **`generate_video.py`**: Create videos from image sequences
- **`download_models.py`**: Download required models from HuggingFace
- **`run_api.py`**: Start the FastAPI server

## Data Flow

### Training Flow
1. Organize data in `data/` directories
2. Run `scripts/train_lora.py`
3. LoRA weights saved to `models/lora_checkpoints/`

### Inference Flow
1. Load base model + ControlNet + LoRA (optional)
2. Input: prompt + pose image (+ attire, character, background)
3. Generate image via `SDXLImageGenerator`
4. Output: 1024x1024 image (5-8 seconds)

### Video Generation Flow
1. Collect generated images
2. Process with `VideoGenerator`
3. Add transitions/effects (optional)
4. Output: MP4 video file

## Model Storage

Models are stored in `models/` (gitignored):
- **Base models**: ~6-7 GB each
- **ControlNet**: ~2-3 GB
- **VAE**: ~300 MB
- **LoRA checkpoints**: ~50-200 MB each

Total storage needed: ~100 GB for all models.

## Output Structure

Generated content stored in `outputs/`:
- **`outputs/images/`**: Generated images (PNG)
- **`outputs/videos/`**: Generated videos (MP4)
- **`outputs/api/`**: API-generated content

## Dependencies

See `requirements.txt` for complete list. Key dependencies:
- **torch**: PyTorch for model operations
- **diffusers**: HuggingFace Diffusers library
- **transformers**: Transformer models
- **peft**: Parameter-Efficient Fine-Tuning (LoRA)
- **controlnet-aux**: ControlNet utilities
- **imageio/moviepy**: Video processing

## Development Workflow

1. **Setup**: Install dependencies, download models
2. **Prepare Data**: Organize training data
3. **Fine-tune**: Train LoRA weights (optional)
4. **Generate**: Create images with trained model
5. **Video**: Generate videos from images
6. **Deploy**: Use API server for production

## Extension Points

The codebase is designed for extension:

1. **Additional ControlNet types**: Add in `generator.py`
2. **IP-Adapter integration**: For better character/attire conditioning
3. **Custom training loops**: Extend `train_lora.py`
4. **Batch processing**: Enhance API server
5. **Additional video effects**: Extend `video_generator.py`

