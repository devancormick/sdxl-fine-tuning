"""Data utilities for loading and preparing training data."""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset


class MultiImageDataset(Dataset):
    """Dataset for handling multiple image inputs (poses, attire, character, background)."""
    
    def __init__(
        self,
        data_dir: str,
        caption_file: Optional[str] = None,
        pose_dir: str = "poses",
        attire_dir: str = "attire",
        character_dir: str = "characters",
        background_dir: str = "backgrounds",
        resolution: int = 1024,
        transform=None,
        prompt_template: str = "a professional portrait, high quality, detailed"
    ):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.transform = transform
        self.prompt_template = prompt_template
        
        # Load captions if provided
        self.captions = {}
        if caption_file:
            caption_path = self.data_dir / caption_file
            if caption_path.exists():
                with open(caption_path, 'r') as f:
                    self.captions = json.load(f)
        
        # Load image paths
        self.pose_paths = sorted(list((self.data_dir / pose_dir).glob("*.png")) + 
                                 list((self.data_dir / pose_dir).glob("*.jpg")))
        self.attire_paths = sorted(list((self.data_dir / attire_dir).glob("*.png")) + 
                                   list((self.data_dir / attire_dir).glob("*.jpg")))
        self.character_paths = sorted(list((self.data_dir / character_dir).glob("*.png")) + 
                                      list((self.data_dir / character_dir).glob("*.jpg")))
        self.background_paths = sorted(list((self.data_dir / background_dir).glob("*.png")) + 
                                       list((self.data_dir / background_dir).glob("*.jpg")))
        
        # Create combinations or use single references
        self.samples = self._create_samples()
    
    def _create_samples(self) -> List[Dict]:
        """Create training samples from available images."""
        samples = []
        
        # If we have matching pairs, use them; otherwise create combinations
        max_samples = max(
            len(self.pose_paths),
            len(self.attire_paths),
            len(self.character_paths),
            len(self.background_paths)
        )
        
        for i in range(max_samples):
            pose_path = self.pose_paths[i % len(self.pose_paths)] if self.pose_paths else None
            caption = None
            
            # Try to get caption for pose image
            if pose_path and self.captions:
                rel_path = str(pose_path.relative_to(self.data_dir))
                caption = self.captions.get(rel_path)
            
            # Use default prompt template if no caption
            if not caption:
                caption = self.prompt_template
            
            sample = {
                "pose": pose_path,
                "attire": self.attire_paths[i % len(self.attire_paths)] if self.attire_paths else None,
                "character": self.character_paths[i % len(self.character_paths)] if self.character_paths else None,
                "background": self.background_paths[i % len(self.background_paths)] if self.background_paths else None,
                "caption": caption,
            }
            samples.append(sample)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load target image (use pose as target if available)
        target_image = None
        if sample["pose"] and sample["pose"].exists():
            target_image = Image.open(sample["pose"]).convert("RGB")
            target_image = target_image.resize((self.resolution, self.resolution), Image.LANCZOS)
        
        return {
            "pixel_values": target_image,
            "prompt": sample["caption"],
        }


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: str):
    """Save configuration to YAML file."""
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def prepare_captions(data_dir: str, output_file: str = "captions.json"):
    """Prepare caption files for training data."""
    import json
    
    captions = {}
    data_path = Path(data_dir)
    
    # Scan all image directories
    for subdir in ["poses", "attire", "characters", "backgrounds"]:
        subdir_path = data_path / subdir
        if subdir_path.exists():
            for img_path in subdir_path.glob("*.png"):
                # Extract caption from filename or use default
                caption = img_path.stem.replace("_", " ")
                captions[str(img_path.relative_to(data_path))] = caption
    
    # Save captions
    output_path = data_path / output_file
    with open(output_path, 'w') as f:
        json.dump(captions, f, indent=2)
    
    return captions

