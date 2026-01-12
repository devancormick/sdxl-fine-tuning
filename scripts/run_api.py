#!/usr/bin/env python
"""Run the SDXL API server."""

import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Change to src directory for imports
os.chdir(project_root / "src")

import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting SDXL API server on {host}:{port}")
    print("Loading model (this may take a while on first request)...")
    
    uvicorn.run("api.server:app", host=host, port=port, reload=False)

