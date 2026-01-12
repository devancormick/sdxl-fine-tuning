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

