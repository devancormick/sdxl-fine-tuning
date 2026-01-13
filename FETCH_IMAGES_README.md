# Image Fetching Script

This script allows you to download images from free online sources for SDXL fine-tuning training data.

## Features

- **LoremPicsum Integration**: Downloads real photos without requiring an API key (default)
- **Pexels API Support**: Optional integration with Pexels API (requires free API key for better results)
- **Multiple Categories**: Supports fetching images for poses, attire, characters, and backgrounds
- **Customizable**: Adjust image dimensions, counts, and search terms

## Quick Start

### Basic Usage (No API Key Required)

```bash
# Fetch 10 images for each category (poses, attire, characters, backgrounds)
python scripts/fetch_images.py --category all --count 10

# Fetch images for a specific category
python scripts/fetch_images.py --category backgrounds --count 20

# Fetch with custom dimensions
python scripts/fetch_images.py --category poses --count 15 --width 1024 --height 1024
```

### Using Pexels API (Recommended for Better Results)

1. Get a free API key from [Pexels API](https://www.pexels.com/api/)
2. Use the API key:

```bash
# Set environment variable (optional)
export PEXELS_API_KEY="your_api_key_here"

# Or pass as argument
python scripts/fetch_images.py \
    --category all \
    --count 20 \
    --use-pexels \
    --pexels-api-key "your_api_key_here"
```

## Options

- `--data-dir`: Directory to save images (default: `./data`)
- `--category`: Category to fetch - `poses`, `attire`, `characters`, `backgrounds`, or `all` (default: `all`)
- `--count`: Number of images per category (default: 10)
- `--width`: Image width in pixels (default: 1024)
- `--height`: Image height in pixels (default: 1024)
- `--pexels-api-key`: Pexels API key (optional)
- `--use-pexels`: Enable Pexels API (requires API key)
- `--search-terms`: Custom search terms (overrides defaults)

## Default Search Terms

- **Poses**: person standing, person posing, portrait pose, human pose, full body
- **Attire**: fashion clothing, outfit, dress, suit, clothing
- **Characters**: portrait, face, person, character
- **Backgrounds**: landscape, nature, city, background, scenery

## Examples

```bash
# Fetch 50 background images
python scripts/fetch_images.py --category backgrounds --count 50

# Fetch custom search terms
python scripts/fetch_images.py \
    --category poses \
    --count 30 \
    --search-terms "yoga pose" "dance pose" "athletic pose"

# Fetch with Pexels API
python scripts/fetch_images.py \
    --category all \
    --count 25 \
    --use-pexels \
    --pexels-api-key "$PEXELS_API_KEY"
```

## Output

Images are saved to:
- `data/poses/poses_001.jpg`, `poses_002.jpg`, etc.
- `data/attire/attire_001.jpg`, `attire_002.jpg`, etc.
- `data/characters/characters_001.jpg`, `characters_002.jpg`, etc.
- `data/backgrounds/backgrounds_001.jpg`, `backgrounds_002.jpg`, etc.

## Next Steps

After fetching images:

1. **Review the downloaded images** - Remove any that don't match your requirements
2. **Add more images** - Fetch additional images or add your own
3. **Run training**:
   ```bash
   python scripts/train_lora.py --config config/training_config.yaml --data_dir data
   ```
4. **Generate images**:
   ```bash
   python scripts/generate_images.py \
       --prompt "your prompt" \
       --pose data/poses/poses_001.jpg \
       --output outputs/images/
   ```

## Notes

- LoremPicsum provides random photos (no search capability, but no API key needed)
- Pexels API offers better search results but requires a free API key
- Images are automatically resized to specified dimensions
- Rate limiting is implemented to avoid overwhelming servers
- All images are saved as JPEG format
