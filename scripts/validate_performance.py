#!/usr/bin/env python
"""Validation script to verify 5-8s generation time and system performance."""

import argparse
import sys
import time
import statistics
from pathlib import Path
from typing import List, Tuple
import torch

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from inference.generator import SDXLImageGenerator
from utils.data_utils import load_config


def validate_generation_time(
    generator: SDXLImageGenerator,
    prompt: str = "a professional portrait, high quality, detailed",
    num_runs: int = 5,
    target_min: float = 5.0,
    target_max: float = 8.0,
    pose_image: str = None,
) -> Tuple[List[float], dict]:
    """
    Validate generation time meets target requirements.
    
    Args:
        generator: Initialized SDXLImageGenerator
        prompt: Test prompt
        num_runs: Number of test runs
        target_min: Minimum acceptable generation time (seconds)
        target_max: Maximum acceptable generation time (seconds)
        pose_image: Optional pose image path
    
    Returns:
        Tuple of (generation_times, statistics_dict)
    """
    print(f"\n{'='*60}")
    print(f"Performance Validation Test")
    print(f"{'='*60}")
    print(f"Target: {target_min}-{target_max} seconds per image")
    print(f"Running {num_runs} test generations...")
    print(f"{'='*60}\n")
    
    generation_times = []
    
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}...", end=" ", flush=True)
        
        try:
            image, gen_time = generator.generate(
                prompt=prompt,
                pose_image=pose_image,
                fast_mode=True,  # Use fast mode for production
                seed=42 + i,  # Different seed each run
            )
            
            generation_times.append(gen_time)
            status = "✓" if target_min <= gen_time <= target_max else "⚠"
            print(f"{status} {gen_time:.2f}s")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    # Calculate statistics
    if not generation_times:
        raise ValueError("No successful generations completed")
    
    stats = {
        "mean": statistics.mean(generation_times),
        "median": statistics.median(generation_times),
        "min": min(generation_times),
        "max": max(generation_times),
        "stdev": statistics.stdev(generation_times) if len(generation_times) > 1 else 0.0,
        "in_target": sum(1 for t in generation_times if target_min <= t <= target_max),
        "total_runs": len(generation_times),
    }
    
    return generation_times, stats


def print_results(stats: dict, target_min: float, target_max: float):
    """Print validation results."""
    print(f"\n{'='*60}")
    print(f"Validation Results")
    print(f"{'='*60}")
    print(f"Mean generation time:     {stats['mean']:.2f}s")
    print(f"Median generation time:   {stats['median']:.2f}s")
    print(f"Min generation time:      {stats['min']:.2f}s")
    print(f"Max generation time:      {stats['max']:.2f}s")
    print(f"Standard deviation:       {stats['stdev']:.2f}s")
    print(f"\nTarget range: {target_min}-{target_max}s")
    print(f"Runs in target: {stats['in_target']}/{stats['total_runs']}")
    
    success_rate = (stats['in_target'] / stats['total_runs']) * 100
    print(f"Success rate: {success_rate:.1f}%")
    
    print(f"\n{'='*60}")
    
    # Overall assessment
    if stats['mean'] >= target_min and stats['mean'] <= target_max:
        print("✓ VALIDATION PASSED: Mean generation time is within target range")
        return True
    elif stats['mean'] < target_min:
        print("⚠ VALIDATION WARNING: Generation is faster than target (may indicate quality issues)")
        return True  # Still pass, but warn
    else:
        print("✗ VALIDATION FAILED: Generation time exceeds target range")
        print(f"  Consider reducing steps or optimizing further")
        return False


def check_system_resources():
    """Check system resources."""
    print(f"\n{'='*60}")
    print(f"System Resources")
    print(f"{'='*60}")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"CUDA reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Check xformers
    try:
        import xformers
        print(f"xformers available: ✓")
    except ImportError:
        print(f"xformers available: ✗ (recommended for performance)")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Validate generation performance")
    parser.add_argument("--config", type=str, default="config/inference_config.yaml", help="Config file")
    parser.add_argument("--model-config", type=str, default="config/model_config.yaml", help="Model config file")
    parser.add_argument("--prompt", type=str, default="a professional portrait, high quality, detailed",
                       help="Test prompt")
    parser.add_argument("--pose", type=str, help="Pose image path")
    parser.add_argument("--runs", type=int, default=5, help="Number of test runs")
    parser.add_argument("--target-min", type=float, default=5.0, help="Minimum target time (seconds)")
    parser.add_argument("--target-max", type=float, default=8.0, help="Maximum target time (seconds)")
    parser.add_argument("--skip-system-check", action="store_true", help="Skip system resource check")
    
    args = parser.parse_args()
    
    # System check
    if not args.skip_system_check:
        check_system_resources()
    
    # Load config
    config = load_config(args.config) if Path(args.config).exists() else {}
    model_config = config.get("model", {})
    
    # Initialize generator
    print("Initializing generator...")
    generator = SDXLImageGenerator(
        base_model_path=model_config.get("base_model"),
        controlnet_model_path=model_config.get("controlnet_model"),
        lora_weights_path=model_config.get("lora_weights"),
        vae_path=model_config.get("vae_model"),
        device=config.get("device", "cuda"),
        dtype=torch.float16,
        enable_optimizations=True,
        model_config_path=args.model_config if Path(args.model_config).exists() else None,
    )
    
    # Run validation
    generation_times, stats = validate_generation_time(
        generator=generator,
        prompt=args.prompt,
        num_runs=args.runs,
        target_min=args.target_min,
        target_max=args.target_max,
        pose_image=args.pose,
    )
    
    # Print results
    passed = print_results(stats, args.target_min, args.target_max)
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
