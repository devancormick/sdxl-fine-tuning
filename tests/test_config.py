"""Tests for configuration files."""

import unittest
from pathlib import Path
import yaml
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.data_utils import load_config


class TestConfig(unittest.TestCase):
    """Test configuration loading."""
    
    def test_load_training_config(self):
        """Test loading training config."""
        config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
        
        if config_path.exists():
            config = load_config(str(config_path))
            
            self.assertIsInstance(config, dict)
            self.assertIn("model", config)
            self.assertIn("training", config)
            self.assertIn("lora", config)
    
    def test_load_inference_config(self):
        """Test loading inference config."""
        config_path = Path(__file__).parent.parent / "config" / "inference_config.yaml"
        
        if config_path.exists():
            config = load_config(str(config_path))
            
            self.assertIsInstance(config, dict)
            self.assertIn("model", config)
            self.assertIn("generation", config)
    
    def test_load_model_config(self):
        """Test loading model config."""
        config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
        
        if config_path.exists():
            config = load_config(str(config_path))
            
            self.assertIsInstance(config, dict)
            self.assertIn("base_models", config)
            self.assertIn("controlnet_models", config)


if __name__ == "__main__":
    unittest.main()

