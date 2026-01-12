#!/usr/bin/env python
"""Training script for LoRA fine-tuning."""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from training.train_lora import main

if __name__ == "__main__":
    main()

