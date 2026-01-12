"""Tests for data utilities."""

import unittest
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.data_utils import prepare_captions, MultiImageDataset


class TestDataUtils(unittest.TestCase):
    """Test data utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.pose_dir = self.temp_dir / "poses"
        self.pose_dir.mkdir(parents=True)
        
        # Create a test image
        self.test_image = Image.new("RGB", (1024, 1024), color="red")
        self.test_image_path = self.pose_dir / "test_pose.png"
        self.test_image.save(self.test_image_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_prepare_captions(self):
        """Test caption preparation."""
        captions = prepare_captions(str(self.temp_dir))
        
        self.assertIsInstance(captions, dict)
        self.assertGreater(len(captions), 0)
        
        # Check that test image has a caption
        caption_key = f"poses/{self.test_image_path.name}"
        self.assertIn(caption_key, captions)
    
    def test_multi_image_dataset(self):
        """Test MultiImageDataset."""
        dataset = MultiImageDataset(
            str(self.temp_dir),
            resolution=512,
        )
        
        self.assertGreater(len(dataset), 0)
        
        # Get a sample
        sample = dataset[0]
        self.assertIsInstance(sample, dict)


if __name__ == "__main__":
    unittest.main()

