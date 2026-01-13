#!/usr/bin/env python3
"""
Script to fetch images from free online sources for SDXL fine-tuning.
Supports:
- LoremPicsum (real photos, no API key needed) - Default
- Pexels API (free API key required, more reliable) - Optional
- Unsplash API (free API key required) - Optional

Note: For best results, use Pexels API with a free key from https://www.pexels.com/api/
"""

import os
import sys
import argparse
import requests
import time
from pathlib import Path
from typing import List, Optional
from PIL import Image
from io import BytesIO
from tqdm import tqdm


# LoremPicsum - Random photos (no key needed, reliable)
LOREM_PICSUM_URL = "https://picsum.photos/{width}/{height}?random={seed}"

# Pixabay API (no key needed for basic usage, but limited)
# Note: Pixabay requires API key for production, but has free tier

# Search terms for different categories
DEFAULT_SEARCH_TERMS = {
    "poses": ["person standing", "person posing", "portrait pose", "human pose", "full body"],
    "attire": ["fashion clothing", "outfit", "dress", "suit", "clothing"],
    "characters": ["portrait", "face", "person", "character"],
    "backgrounds": ["landscape", "nature", "city", "background", "scenery"]
}


def fetch_lorempicsum_image(width: int = 1024, height: int = 1024, seed: Optional[int] = None, max_retries: int = 3) -> Optional[Image.Image]:
    """Fetch a random image from LoremPicsum (real photos, no API key needed)."""
    if seed is None:
        import random
        seed = random.randint(1, 10000)
    
    url = LOREM_PICSUM_URL.format(width=width, height=height, seed=seed)
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=15, allow_redirects=True)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content))
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            return img
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None
    
    return None


def fetch_pexels_image(api_key: Optional[str], query: str, width: int = 1024, height: int = 1024) -> Optional[Image.Image]:
    """Fetch a single image from Pexels API (requires free API key)."""
    if not api_key:
        return None
    
    try:
        # Search for photos
        search_url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": api_key}
        params = {"query": query, "per_page": 1, "orientation": "portrait" if height > width else "landscape"}
        
        response = requests.get(search_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data.get("photos") and len(data["photos"]) > 0:
            photo = data["photos"][0]
            image_url = photo["src"]["original"]  # Get highest quality
            
            # Download the image
            img_response = requests.get(image_url, timeout=10)
            img_response.raise_for_status()
            
            img = Image.open(BytesIO(img_response.content))
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Resize to desired dimensions
            img = img.resize((width, height), Image.LANCZOS)
            
            return img
    except Exception as e:
        print(f"Pexels API error for '{query}': {e}")
        return None
    
    return None


def fetch_image_from_url(image_url: str, width: int = 1024, height: int = 1024) -> Optional[Image.Image]:
    """Fetch an image from a direct URL."""
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        img = img.resize((width, height), Image.LANCZOS)
        return img
    except Exception as e:
        print(f"Failed to fetch image from URL: {e}")
        return None


def save_image(img: Image.Image, output_path: Path, quality: int = 95):
    """Save image to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as JPEG
    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        img.save(output_path, "JPEG", quality=quality, optimize=True)
    else:
        img.save(output_path, "PNG", optimize=True)


def fetch_category_images(
    category: str,
    output_dir: Path,
    count: int,
    search_terms: List[str],
    width: int = 1024,
    height: int = 1024,
    use_pexels: bool = False,
    pexels_api_key: Optional[str] = None,
    start_index: int = 0
) -> int:
    """Fetch images for a specific category."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    query_index = 0
    
    print(f"\nFetching {count} images for category: {category}")
    print(f"Search terms: {', '.join(search_terms)}")
    
    for i in tqdm(range(count), desc=f"Fetching {category}"):
        # Cycle through search terms
        query = search_terms[query_index % len(search_terms)]
        query_index += 1
        
        # Try to fetch image
        img = None
        
        # Try Pexels first if enabled
        if use_pexels and pexels_api_key:
            img = fetch_pexels_image(pexels_api_key, query, width, height)
            if img:
                time.sleep(0.5)  # Rate limiting
        
        # Fallback to LoremPicsum if Pexels didn't work (real photos, no key needed)
        if not img:
            import random
            img = fetch_lorempicsum_image(width, height, seed=random.randint(1, 10000))
            if img:
                time.sleep(0.5)  # Rate limiting
        
        if img:
            # Save image
            image_path = output_dir / f"{category}_{start_index + saved_count + 1:03d}.jpg"
            save_image(img, image_path)
            saved_count += 1
        else:
            print(f"\nWarning: Failed to fetch image {i+1} for {category}")
    
    print(f"✓ Saved {saved_count}/{count} images to {output_dir}")
    return saved_count


def main():
    parser = argparse.ArgumentParser(
        description="Fetch images from free online sources for SDXL fine-tuning"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to save images (default: ./data)"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["poses", "attire", "characters", "backgrounds", "all"],
        default="all",
        help="Category to fetch (default: all)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of images to fetch per category (default: 10)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (default: 1024)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (default: 1024)"
    )
    parser.add_argument(
        "--pexels-api-key",
        type=str,
        default=None,
        help="Pexels API key (optional, get free key from https://www.pexels.com/api/)"
    )
    parser.add_argument(
        "--use-pexels",
        action="store_true",
        help="Use Pexels API (requires --pexels-api-key)"
    )
    parser.add_argument(
        "--search-terms",
        type=str,
        nargs="+",
        default=None,
        help="Custom search terms (overrides defaults)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Categories to process
    categories = ["poses", "attire", "characters", "backgrounds"] if args.category == "all" else [args.category]
    
    # Check if Pexels API key is provided when use_pexels is enabled
    if args.use_pexels and not args.pexels_api_key:
        print("Warning: --use-pexels specified but no --pexels-api-key provided. Using Unsplash only.")
        args.use_pexels = False
    
    # Also check environment variable for Pexels API key
    if not args.pexels_api_key:
        args.pexels_api_key = os.getenv("PEXELS_API_KEY")
    
    total_saved = 0
    
    for category in categories:
        # Get search terms
        if args.search_terms:
            search_terms = args.search_terms
        else:
            search_terms = DEFAULT_SEARCH_TERMS.get(category, [category])
        
        output_dir = data_dir / category
        
        saved = fetch_category_images(
            category=category,
            output_dir=output_dir,
            count=args.count,
            search_terms=search_terms,
            width=args.width,
            height=args.height,
            use_pexels=args.use_pexels,
            pexels_api_key=args.pexels_api_key
        )
        
        total_saved += saved
    
    print(f"\n{'='*60}")
    print(f"Total images saved: {total_saved}")
    print(f"Images saved to: {data_dir}")
    print(f"{'='*60}")
    
    if total_saved > 0:
        print("\nNext steps:")
        print("1. Review the downloaded images")
        print("2. Optionally filter/clean the images")
        print("3. Run training: python scripts/train_lora.py")
        print("4. Generate images: python scripts/generate_images.py")


if __name__ == "__main__":
    main()
