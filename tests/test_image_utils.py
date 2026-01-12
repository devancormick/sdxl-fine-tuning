"""Tests for image utilities."""

import unittest
from pathlib import Path
from PIL import Image
import torch
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.image_utils import (
    pil_to_tensor,
    tensor_to_pil,
    resize_image,
    preprocess_image_for_controlnet,
)


class TestImageUtils(unittest.TestCase):
    """Test image utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_image = Image.new("RGB", (512, 512), color="blue")
    
    def test_pil_to_tensor(self):
        """Test PIL to tensor conversion."""
        tensor = pil_to_tensor(self.test_image)
        
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (3, 512, 512))
        self.assertTrue(torch.all(tensor >= 0) and torch.all(tensor <= 1))
    
    def test_tensor_to_pil(self):
        """Test tensor to PIL conversion."""
        tensor = pil_to_tensor(self.test_image)
        pil_image = tensor_to_pil(tensor)
        
        self.assertIsInstance(pil_image, Image.Image)
        self.assertEqual(pil_image.size, (512, 512))
    
    def test_resize_image(self):
        """Test image resizing."""
        resized = resize_image(self.test_image, (1024, 1024))
        
        self.assertEqual(resized.size, (1024, 1024))
        self.assertIsInstance(resized, Image.Image)
    
    def test_preprocess_image_for_controlnet(self):
        """Test ControlNet image preprocessing."""
        tensor = preprocess_image_for_controlnet(self.test_image, (1024, 1024))
        
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (1, 3, 1024, 1024))
        # Check normalization range [-1, 1]
        self.assertTrue(torch.all(tensor >= -1) and torch.all(tensor <= 1))


if __name__ == "__main__":
    unittest.main()

