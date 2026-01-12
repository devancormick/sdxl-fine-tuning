#!/usr/bin/env python
"""Validate and prepare training data."""

import argparse
import sys
from pathlib import Path
from PIL import Image
import json

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.data_utils import prepare_captions


def validate_images(directory: Path, min_resolution: int = 512) -> dict:
    """Validate images in a directory."""
    stats = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "errors": [],
        "resolutions": [],
    }
    
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = [f for f in directory.iterdir() if f.suffix.lower() in image_extensions]
    
    stats["total"] = len(image_files)
    
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                # Check format
                if img.format not in ["PNG", "JPEG", "WEBP"]:
                    stats["errors"].append(f"{img_path.name}: Invalid format {img.format}")
                    stats["invalid"] += 1
                    continue
                
                # Check resolution
                width, height = img.size
                min_dim = min(width, height)
                if min_dim < min_resolution:
                    stats["errors"].append(
                        f"{img_path.name}: Resolution {width}x{height} below minimum {min_resolution}"
                    )
                    stats["invalid"] += 1
                    continue
                
                stats["valid"] += 1
                stats["resolutions"].append((width, height))
        
        except Exception as e:
            stats["errors"].append(f"{img_path.name}: {str(e)}")
            stats["invalid"] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Validate training data")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--min-resolution", type=int, default=512, help="Minimum resolution")
    parser.add_argument("--generate-captions", action="store_true", help="Generate caption file")
    parser.add_argument("--output", type=str, help="Output validation report JSON file")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        sys.exit(1)
    
    print("Validating training data...\n")
    
    categories = ["poses", "attire", "characters", "backgrounds"]
    all_stats = {}
    
    for category in categories:
        category_dir = data_dir / category
        if category_dir.exists():
            print(f"Validating {category}...")
            stats = validate_images(category_dir, args.min_resolution)
            all_stats[category] = stats
            
            print(f"  Total: {stats['total']}")
            print(f"  Valid: {stats['valid']}")
            print(f"  Invalid: {stats['invalid']}")
            if stats["errors"]:
                print(f"  Errors: {len(stats['errors'])}")
                for error in stats["errors"][:5]:  # Show first 5 errors
                    print(f"    - {error}")
                if len(stats["errors"]) > 5:
                    print(f"    ... and {len(stats['errors']) - 5} more")
            print()
        else:
            print(f"Skipping {category} (directory does not exist)\n")
    
    # Generate captions if requested
    if args.generate_captions:
        print("Generating captions...")
        captions = prepare_captions(str(data_dir))
        print(f"Generated captions for {len(captions)} images")
        print()
    
    # Save report if requested
    if args.output:
        report = {
            "validation": all_stats,
            "summary": {
                "total_images": sum(s["total"] for s in all_stats.values()),
                "valid_images": sum(s["valid"] for s in all_stats.values()),
                "invalid_images": sum(s["invalid"] for s in all_stats.values()),
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Validation report saved to {args.output}")
    
    # Exit code based on validation results
    total_invalid = sum(s["invalid"] for s in all_stats.values())
    if total_invalid > 0:
        print(f"Warning: {total_invalid} invalid images found")
        sys.exit(1)
    
    print("All images are valid!")


if __name__ == "__main__":
    main()

