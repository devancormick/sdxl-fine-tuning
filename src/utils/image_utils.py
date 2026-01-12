"""Image processing utilities."""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import Union, Tuple


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to tensor."""
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.permute(1, 2, 0).cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return Image.fromarray(tensor)


def resize_image(image: Image.Image, size: Tuple[int, int], resample=Image.LANCZOS) -> Image.Image:
    """Resize image while maintaining aspect ratio."""
    image.thumbnail(size, resample)
    # Create new image with target size and paste resized image
    new_image = Image.new('RGB', size, (0, 0, 0))
    new_image.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return new_image


def preprocess_image_for_controlnet(image: Image.Image, target_size: Tuple[int, int] = (1024, 1024)) -> torch.Tensor:
    """Preprocess image for ControlNet input."""
    # Convert to numpy array
    image = image.convert("RGB")
    image = image.resize(target_size, Image.LANCZOS)
    image_np = np.array(image)
    
    # Normalize to [-1, 1]
    image_np = image_np.astype(np.float32) / 255.0
    image_np = image_np * 2.0 - 1.0
    
    # Convert to tensor [C, H, W]
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    return image_tensor.unsqueeze(0)  # Add batch dimension


def extract_pose_keypoints(image: Image.Image) -> np.ndarray:
    """Extract pose keypoints from image using OpenPose-like preprocessing."""
    # This is a placeholder - in production, you'd use actual OpenPose or MediaPipe
    # For now, we'll just return the image as-is for ControlNet
    image_np = np.array(image.convert("RGB"))
    return image_np


def blend_reference_images(images: list, blend_weights: Optional[list] = None) -> Image.Image:
    """Blend multiple reference images together with optional weights."""
    if not images:
        raise ValueError("No images provided")
    
    if len(images) == 1:
        return images[0]
    
    # Ensure all images are the same size
    target_size = images[0].size
    images_resized = []
    for img in images:
        if isinstance(img, str):
            from pathlib import Path
            img = Image.open(img)
        if img.size != target_size:
            img = resize_image(img, target_size)
        images_resized.append(np.array(img.convert("RGB")).astype(np.float32))
    
    # Default weights: equal weight for all images
    if blend_weights is None:
        blend_weights = [1.0 / len(images)] * len(images)
    
    # Normalize weights
    total_weight = sum(blend_weights)
    blend_weights = [w / total_weight for w in blend_weights]
    
    # Blend images
    blended = np.zeros_like(images_resized[0])
    for img, weight in zip(images_resized, blend_weights):
        blended += img * weight
    
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def enhance_prompt_with_references(
    base_prompt: str,
    character_image: Optional[Image.Image] = None,
    attire_image: Optional[Image.Image] = None,
    background_image: Optional[Image.Image] = None,
) -> str:
    """Enhance a prompt with reference image descriptions."""
    prompt_parts = [base_prompt]
    
    if character_image:
        prompt_parts.append("matching character style and features")
    
    if attire_image:
        prompt_parts.append("matching attire and clothing style")
    
    if background_image:
        prompt_parts.append("matching background and setting")
    
    return ", ".join(prompt_parts)
