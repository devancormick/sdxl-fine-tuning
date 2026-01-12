# Production Deployment Guide

## Overview

This guide covers deploying the SDXL fine-tuning pipeline in a production environment with the goal of 5-8 second image generation times.

## System Requirements

### Minimum
- GPU: NVIDIA GPU with 12GB+ VRAM (RTX 3060, RTX 3070, etc.)
- CPU: 8+ cores recommended
- RAM: 32GB+ recommended
- Storage: 100GB+ SSD (for models and cache)

### Recommended (for 5-8s generation)
- GPU: NVIDIA GPU with 16GB+ VRAM (RTX 3090, RTX 4090, A100, etc.)
- CPU: 12+ cores
- RAM: 64GB+
- Storage: NVMe SSD

## Optimization Strategies

### 1. Model Loading

**Keep models in memory:**
- Load models once at startup
- Use a singleton pattern or persistent service
- Avoid reloading between requests

```python
# Example: Model service
class ModelService:
    _instance = None
    _generator = None
    
    @classmethod
    def get_generator(cls):
        if cls._generator is None:
            cls._generator = SDXLImageGenerator(...)
        return cls._generator
```

### 2. Inference Optimizations

**Enable all optimizations:**
- XFormers memory efficient attention
- VAE slicing
- Attention slicing
- Torch.compile (PyTorch 2.0+)
- FP16 precision

**Reduce inference steps:**
- 20-25 steps for speed (target: 5-8s)
- 30-40 steps for quality
- Use EulerAncestralDiscreteScheduler (faster)

### 3. Batch Processing

For production, batch requests when possible:

```python
def generate_batch(prompts, pose_images):
    # Process multiple images in parallel
    # Adjust batch size based on GPU memory
    pass
```

### 4. Caching

Cache frequently used:
- Pose images (preprocessed)
- ControlNet outputs
- Common prompts

## API Deployment

### FastAPI Example

```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import tempfile

app = FastAPI()
generator = ModelService.get_generator()

@app.post("/generate")
async def generate_image(
    prompt: str,
    pose: UploadFile = File(...),
    # ... other params
):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await pose.read())
        pose_path = tmp.name
    
    # Generate
    image, gen_time = generator.generate(
        prompt=prompt,
        pose_image=pose_path,
        # ... other params
    )
    
    # Save and return
    output_path = save_image(image)
    return FileResponse(output_path)
```

### Docker Deployment

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Run API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sdxl-api
spec:
  replicas: 1  # Scale based on GPU availability
  template:
    spec:
      containers:
      - name: sdxl-api
        image: your-registry/sdxl-api:latest
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            nvidia.com/gpu: 1
```

## Monitoring

### Metrics to Track

1. **Generation time** (target: 5-8s)
2. **GPU utilization**
3. **Memory usage**
4. **Queue length** (if using async processing)
5. **Error rate**

### Logging

```python
import logging
import time

logger = logging.getLogger(__name__)

def generate_with_logging(prompt, **kwargs):
    start = time.time()
    try:
        image, gen_time = generator.generate(prompt, **kwargs)
        logger.info(f"Generated image in {gen_time:.2f}s")
        return image
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise
```

## Scaling Strategies

### Horizontal Scaling
- Multiple GPU instances
- Load balancer
- Shared model storage (NFS, S3)

### Vertical Scaling
- Larger GPUs (A100, H100)
- Multiple GPUs per instance
- Model parallelism

### Queue-Based Processing
- Celery + Redis/RabbitMQ
- Process requests asynchronously
- Return job IDs, poll for results

## Performance Benchmarks

### Expected Performance (RTX 3090)

| Resolution | Steps | Time | Quality |
|------------|-------|------|---------|
| 1024x1024  | 20    | ~4s  | Good    |
| 1024x1024  | 25    | ~6s  | Better  |
| 1024x1024  | 30    | ~8s  | Best    |

### Optimization Checklist

- [ ] XFormers enabled
- [ ] VAE slicing enabled
- [ ] FP16 precision
- [ ] Torch.compile (if PyTorch 2.0+)
- [ ] Models preloaded
- [ ] Inference steps optimized (20-25)
- [ ] Batch processing (if applicable)
- [ ] Caching enabled
- [ ] Monitoring in place

## Cost Estimation

### AWS EC2 (example)

- **g4dn.xlarge** (1x T4, 16GB VRAM): ~$0.526/hour
- **g5.2xlarge** (1x A10G, 24GB VRAM): ~$1.008/hour
- **p3.2xlarge** (1x V100, 16GB VRAM): ~$3.06/hour

### Consider
- Reserved instances for steady load
- Spot instances for non-critical workloads
- Auto-scaling based on queue depth

## Security

1. **Input validation**: Validate all inputs
2. **Rate limiting**: Prevent abuse
3. **Authentication**: API keys or OAuth
4. **Resource limits**: Max image size, timeout
5. **Content filtering**: Filter inappropriate prompts

## Maintenance

- Regular model updates
- Monitor disk space (model cache)
- Clear old generated images
- Update dependencies
- Backup configurations

