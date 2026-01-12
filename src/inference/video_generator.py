"""Video generation from image sequences."""

import os
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image
import numpy as np
import imageio
from tqdm import tqdm


class VideoGenerator:
    """Generate videos from image sequences."""
    
    def __init__(self, fps: int = 24, codec: str = "libx264", quality: str = "high"):
        self.fps = fps
        self.codec = codec
        self.quality = quality
    
    def generate_from_images(
        self,
        image_paths: List[str],
        output_path: str,
        fps: Optional[int] = None,
        loop: bool = False,
        transition_frames: int = 0,
    ) -> str:
        """
        Generate video from list of image paths.
        
        Args:
            image_paths: List of paths to images
            output_path: Output video path
            fps: Frames per second (overrides default)
            loop: Whether to loop the video
            transition_frames: Number of transition frames between images
        
        Returns:
            Path to generated video
        """
        fps = fps or self.fps
        
        # Load images
        images = []
        for img_path in tqdm(image_paths, desc="Loading images"):
            if Path(img_path).exists():
                img = Image.open(img_path).convert("RGB")
                images.append(np.array(img))
        
        if not images:
            raise ValueError("No valid images found")
        
        # Ensure all images have the same size
        target_size = images[0].shape[:2][::-1]  # (width, height)
        processed_images = []
        
        for img in tqdm(images, desc="Processing images"):
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize(target_size, Image.LANCZOS)
            processed_images.append(np.array(img_pil))
        
        # Add transitions if requested
        if transition_frames > 0:
            processed_images = self._add_transitions(processed_images, transition_frames)
        
        # Loop if requested
        if loop:
            processed_images = processed_images + processed_images[::-1]
        
        # Generate video
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Writing video to {output_path}...")
        imageio.mimwrite(
            str(output_path),
            processed_images,
            fps=fps,
            codec=self.codec,
            quality=8 if self.quality == "high" else 5,
        )
        
        print(f"Video saved to {output_path}")
        return str(output_path)
    
    def generate_from_directory(
        self,
        image_dir: str,
        output_path: str,
        pattern: str = "*.png",
        sort: bool = True,
        **kwargs
    ) -> str:
        """Generate video from directory of images."""
        image_dir = Path(image_dir)
        image_paths = list(image_dir.glob(pattern))
        
        if sort:
            image_paths = sorted(image_paths)
        
        return self.generate_from_images(
            [str(p) for p in image_paths],
            output_path,
            **kwargs
        )
    
    def _add_transitions(self, images: List[np.ndarray], transition_frames: int) -> List[np.ndarray]:
        """Add crossfade transitions between images."""
        if len(images) <= 1 or transition_frames == 0:
            return images
        
        result = []
        for i in range(len(images)):
            result.append(images[i])
            
            # Add transition to next image
            if i < len(images) - 1:
                for j in range(1, transition_frames + 1):
                    alpha = j / (transition_frames + 1)
                    blended = (images[i] * (1 - alpha) + images[i + 1] * alpha).astype(np.uint8)
                    result.append(blended)
        
        return result
    
    def create_slideshow(
        self,
        image_paths: List[str],
        output_path: str,
        duration_per_image: float = 2.0,
        transition_duration: float = 0.5,
        **kwargs
    ) -> str:
        """Create slideshow with specified duration per image."""
        # Calculate fps needed
        fps = kwargs.get("fps", self.fps)
        frames_per_image = int(duration_per_image * fps)
        transition_frames = int(transition_duration * fps)
        
        # Load and duplicate images
        images = []
        for img_path in image_paths:
            if Path(img_path).exists():
                img = Image.open(img_path).convert("RGB")
                # Add multiple copies of the same image for duration
                for _ in range(frames_per_image):
                    images.append(img_path)
        
        return self.generate_from_images(
            images,
            output_path,
            fps=fps,
            transition_frames=transition_frames,
            **kwargs
        )

