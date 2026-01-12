"""Model utilities for verification and selection."""

import os
from typing import Optional, Dict, Tuple
from huggingface_hub import model_info, HfApi
from pathlib import Path


def verify_model_available(model_path: str, model_type: str = "model") -> Tuple[bool, Optional[str]]:
    """
    Verify if a model is available on HuggingFace or locally.
    
    Args:
        model_path: Model identifier (HuggingFace repo ID or local path)
        model_type: Type of model for logging ("model", "controlnet", "vae")
    
    Returns:
        Tuple of (is_available, error_message)
    """
    # Check if it's a local path
    if Path(model_path).exists() or os.path.isdir(model_path):
        return True, None
    
    # Check if it's a HuggingFace model
    try:
        api = HfApi()
        model_info_result = model_info(model_path, token=None)
        
        if model_info_result:
            return True, None
        else:
            return False, f"{model_type} '{model_path}' not found on HuggingFace Hub"
    
    except Exception as e:
        # If model doesn't exist, check if it's a valid path pattern
        if "/" in model_path and len(model_path.split("/")) == 2:
            # Looks like a HuggingFace repo ID but doesn't exist
            return False, f"{model_type} '{model_path}' not found: {str(e)}"
        else:
            # Might be a local path that doesn't exist yet
            return False, f"{model_type} '{model_path}' not accessible: {str(e)}"


def select_base_model(
    model_config: Dict,
    preferred_model: Optional[str] = None,
    fallback_to_default: bool = True
) -> Tuple[str, str]:
    """
    Select base model with verification and fallback.
    
    Args:
        model_config: Model configuration dictionary
        preferred_model: Preferred model name (e.g., "endgame", "gonzalomo", "sdxl_base")
        fallback_to_default: If True, fallback to SDXL base if preferred model unavailable
    
    Returns:
        Tuple of (model_path, model_name)
    """
    base_models = model_config.get("base_models", {})
    
    # Default fallback
    default_model = base_models.get("sdxl_base", "stabilityai/stable-diffusion-xl-base-1.0")
    default_name = "sdxl_base"
    
    # If no preference, use default
    if not preferred_model:
        return default_model, default_name
    
    # Check preferred model
    preferred_path = base_models.get(preferred_model)
    
    if preferred_path:
        # Verify model is available
        is_available, error = verify_model_available(preferred_path, f"Base model ({preferred_model})")
        
        if is_available:
            print(f"✓ Using {preferred_model} model: {preferred_path}")
            return preferred_path, preferred_model
        else:
            print(f"⚠ {error}")
            if fallback_to_default:
                print(f"⚠ Falling back to default SDXL base model: {default_model}")
                return default_model, default_name
            else:
                raise ValueError(f"Model {preferred_model} not available and fallback disabled")
    else:
        print(f"⚠ Model '{preferred_model}' not found in config. Available models: {list(base_models.keys())}")
        if fallback_to_default:
            print(f"⚠ Using default SDXL base model: {default_model}")
            return default_model, default_name
        else:
            raise ValueError(f"Model {preferred_model} not in config")


def verify_controlnet_model(controlnet_path: str) -> Tuple[bool, Optional[str]]:
    """Verify ControlNet model is available."""
    return verify_model_available(controlnet_path, "ControlNet")


def verify_vae_model(vae_path: str) -> Tuple[bool, Optional[str]]:
    """Verify VAE model is available."""
    return verify_model_available(vae_path, "VAE")


def load_model_config(config_path: str) -> Dict:
    """Load and validate model configuration."""
    from utils.data_utils import load_config
    
    config = load_config(config_path)
    
    # Validate required keys
    if "base_models" not in config:
        raise ValueError("Model config must contain 'base_models' key")
    
    if "controlnet_models" not in config:
        raise ValueError("Model config must contain 'controlnet_models' key")
    
    return config
