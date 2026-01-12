"""FastAPI server for SDXL image generation."""

import os
import sys
from pathlib import Path
from typing import Optional
import tempfile
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from inference.generator import SDXLImageGenerator
from utils.data_utils import load_config
import torch


class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "blurry, low quality, distorted, bad anatomy"
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None


# Global generator instance
generator = None


def get_generator():
    """Get or initialize the generator (singleton pattern)."""
    global generator
    if generator is None:
        config_path = project_root / "config" / "inference_config.yaml"
        config = load_config(str(config_path)) if config_path.exists() else {}
        model_config = config.get("model", {})
        
        generator = SDXLImageGenerator(
            base_model_path=model_config.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0"),
            controlnet_model_path=model_config.get("controlnet_model", "thibaud/controlnet-openpose-sdxl-1.0"),
            lora_weights_path=model_config.get("lora_weights"),
            vae_path=model_config.get("vae_model"),
            device=config.get("device", "cuda"),
            dtype=torch.float16,
            enable_optimizations=True,
        )
    return generator


app = FastAPI(title="SDXL Image Generation API", version="1.0.0")


@app.get("/")
async def root():
    return {"message": "SDXL Image Generation API", "status": "running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        gen = get_generator()
        return {"status": "healthy", "model_loaded": gen is not None}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/generate")
async def generate_image(
    prompt: str = Form(...),
    pose: Optional[UploadFile] = File(None),
    negative_prompt: str = Form("blurry, low quality, distorted, bad anatomy"),
    width: int = Form(1024),
    height: int = Form(1024),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(7.5),
    seed: Optional[int] = Form(None),
):
    """
    Generate image from text prompt and optional pose image.
    
    Returns the generated image as PNG.
    """
    try:
        gen = get_generator()
        
        # Save uploaded pose image temporarily
        pose_path = None
        if pose:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                shutil.copyfileobj(pose.file, tmp)
                pose_path = tmp.name
        
        # Generate image
        image, gen_time = gen.generate(
            prompt=prompt,
            pose_image=pose_path,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        
        # Save to temporary file
        output_dir = project_root / "outputs" / "api"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with tempfile.NamedTemporaryFile(delete=False, dir=output_dir, suffix=".png") as tmp:
            image.save(tmp.name)
            output_path = tmp.name
        
        # Clean up pose temp file
        if pose_path and os.path.exists(pose_path):
            os.unlink(pose_path)
        
        return FileResponse(
            output_path,
            media_type="image/png",
            headers={
                "X-Generation-Time": f"{gen_time:.2f}",
                "X-Prompt": prompt,
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/batch")
async def generate_batch(requests: list[GenerationRequest]):
    """
    Generate multiple images in batch.
    
    Note: This is a placeholder. Implement batching based on GPU memory.
    """
    # TODO: Implement batch processing
    raise HTTPException(status_code=501, detail="Batch processing not yet implemented")


def main():
    """Run the API server."""
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting SDXL API server on {host}:{port}")
    print("Loading model (this may take a while on first request)...")
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

