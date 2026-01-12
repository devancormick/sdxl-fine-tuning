#!/usr/bin/env python
"""Video generation script from image sequences."""

import argparse
import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from inference.video_generator import VideoGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate video from images")
    parser.add_argument("--input", type=str, required=True, help="Input directory or file with image paths")
    parser.add_argument("--output", type=str, required=True, help="Output video path")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--pattern", type=str, default="*.png", help="Image pattern (if input is directory)")
    parser.add_argument("--loop", action="store_true", help="Loop the video")
    parser.add_argument("--transition-frames", type=int, default=0, help="Number of transition frames")
    parser.add_argument("--duration-per-image", type=float, help="Duration per image (slideshow mode)")
    parser.add_argument("--transition-duration", type=float, default=0.5, help="Transition duration (slideshow mode)")
    parser.add_argument("--quality", type=str, default="high", choices=["high", "medium", "low"], help="Video quality")
    
    args = parser.parse_args()
    
    generator = VideoGenerator(fps=args.fps, quality=args.quality)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Read image paths from file
        with open(input_path, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        if args.duration_per_image:
            generator.create_slideshow(
                image_paths,
                args.output,
                duration_per_image=args.duration_per_image,
                transition_duration=args.transition_duration,
                loop=args.loop,
            )
        else:
            generator.generate_from_images(
                image_paths,
                args.output,
                fps=args.fps,
                loop=args.loop,
                transition_frames=args.transition_frames,
            )
    elif input_path.is_dir():
        # Generate from directory
        generator.generate_from_directory(
            str(input_path),
            args.output,
            pattern=args.pattern,
            loop=args.loop,
            transition_frames=args.transition_frames,
        )
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()

