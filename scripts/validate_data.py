#!/usr/bin/env python
"""Validate and prepare training data."""

import argparse
import sys
import hashlib
from pathlib import Path
from PIL import Image
import json
from collections import defaultdict
from typing import Dict, List, Tuple

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.data_utils import prepare_captions


def calculate_image_hash(image_path: Path) -> str:
    """Calculate MD5 hash of image file."""
    hash_md5 = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def validate_images(directory: Path, min_resolution: int = 512, check_duplicates: bool = False) -> dict:
    """Validate images in a directory."""
    stats = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "errors": [],
        "resolutions": [],
        "file_sizes": [],
        "duplicates": [],
        "formats": defaultdict(int),
    }
    
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = [f for f in directory.iterdir() if f.suffix.lower() in image_extensions]
    
    stats["total"] = len(image_files)
    image_hashes = {}
    
    for img_path in image_files:
        try:
            # Check file size
            file_size = img_path.stat().st_size
            stats["file_sizes"].append(file_size)
            
            with Image.open(img_path) as img:
                # Check format
                format_name = img.format or "UNKNOWN"
                stats["formats"][format_name] += 1
                
                if format_name not in ["PNG", "JPEG", "WEBP"]:
                    stats["errors"].append(f"{img_path.name}: Invalid format {format_name}")
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
                
                # Check for duplicates
                if check_duplicates:
                    img_hash = calculate_image_hash(img_path)
                    if img_hash in image_hashes:
                        duplicate_path = image_hashes[img_hash]
                        stats["duplicates"].append({
                            "file": str(img_path.relative_to(directory)),
                            "duplicate_of": str(duplicate_path.relative_to(directory)),
                        })
                        stats["invalid"] += 1
                        continue
                    image_hashes[img_hash] = img_path
                
                stats["valid"] += 1
                stats["resolutions"].append((width, height))
        
        except Exception as e:
            stats["errors"].append(f"{img_path.name}: {str(e)}")
            stats["invalid"] += 1
    
    # Calculate statistics
    if stats["resolutions"]:
        widths, heights = zip(*stats["resolutions"])
        stats["resolution_stats"] = {
            "min_width": min(widths),
            "max_width": max(widths),
            "avg_width": sum(widths) / len(widths),
            "min_height": min(heights),
            "max_height": max(heights),
            "avg_height": sum(heights) / len(heights),
        }
    
    if stats["file_sizes"]:
        stats["file_size_stats"] = {
            "min_size": min(stats["file_sizes"]),
            "max_size": max(stats["file_sizes"]),
            "avg_size": sum(stats["file_sizes"]) / len(stats["file_sizes"]),
            "total_size": sum(stats["file_sizes"]),
        }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Validate training data")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--min-resolution", type=int, default=512, help="Minimum resolution")
    parser.add_argument("--generate-captions", action="store_true", help="Generate caption file")
    parser.add_argument("--check-duplicates", action="store_true", help="Check for duplicate images")
    parser.add_argument("--output", type=str, help="Output validation report JSON file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
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
            stats = validate_images(category_dir, args.min_resolution, args.check_duplicates)
            all_stats[category] = stats
            
            print(f"  Total: {stats['total']}")
            print(f"  Valid: {stats['valid']}")
            print(f"  Invalid: {stats['invalid']}")
            
            if args.verbose or stats.get("resolution_stats"):
                if "resolution_stats" in stats:
                    res_stats = stats["resolution_stats"]
                    print(f"  Resolution: {res_stats['min_width']}x{res_stats['min_height']} - "
                          f"{res_stats['max_width']}x{res_stats['max_height']} "
                          f"(avg: {res_stats['avg_width']:.0f}x{res_stats['avg_height']:.0f})")
                
                if "file_size_stats" in stats:
                    size_stats = stats["file_size_stats"]
                    total_mb = size_stats["total_size"] / (1024 * 1024)
                    avg_mb = size_stats["avg_size"] / (1024 * 1024)
                    print(f"  File size: {total_mb:.2f} MB total (avg: {avg_mb:.2f} MB)")
            
            if stats.get("formats"):
                formats_str = ", ".join([f"{k}: {v}" for k, v in stats["formats"].items()])
                print(f"  Formats: {formats_str}")
            
            if stats["errors"]:
                print(f"  Errors: {len(stats['errors'])}")
                for error in stats["errors"][:5]:  # Show first 5 errors
                    print(f"    - {error}")
                if len(stats["errors"]) > 5:
                    print(f"    ... and {len(stats['errors']) - 5} more")
            
            if stats.get("duplicates"):
                print(f"  Duplicates: {len(stats['duplicates'])}")
                if args.verbose:
                    for dup in stats["duplicates"][:5]:
                        print(f"    - {dup['file']} duplicates {dup['duplicate_of']}")
            
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

