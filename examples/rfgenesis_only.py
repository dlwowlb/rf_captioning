#!/usr/bin/env python3
"""
RF-Genesis Standalone Example
==============================

This example demonstrates how to use RF-Genesis directly
to generate RF Doppler signals from text prompts.

RF-Genesis uses MDM (Motion Diffusion Model) internally to generate motion,
then simulates RF signals through ray tracing.

Usage:
    python examples/rfgenesis_only.py \
        --prompt "a person walking forward" \
        --output-dir output/rf_test

This is equivalent to running RF-Genesis's run.py directly.
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
import time

# ============================================================================
# Path Setup
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RF_GENESIS_DIR = PROJECT_ROOT / "RF-Genesis"

if RF_GENESIS_DIR.exists():
    sys.path.insert(0, str(RF_GENESIS_DIR))


def main():
    parser = argparse.ArgumentParser(description="RF-Genesis standalone example")
    parser.add_argument(
        "-p", "--prompt",
        required=True,
        help="Text prompt for motion/RF generation"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Output directory"
    )
    parser.add_argument(
        "-n", "--name",
        default=None,
        help="Experiment name (optional)"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization"
    )
    parser.add_argument(
        "--no-environment",
        action="store_true",
        help="Skip environment PIR generation"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RF-Genesis Standalone Pipeline")
    print("=" * 60)

    try:
        from genesis.raytracing import pathtracer, signal_generator
        from genesis.object_diffusion import object_diff
        from genesis.visualization import visualize
        print("[OK] RF-Genesis modules loaded")
    except ImportError as e:
        print(f"[ERROR] Failed to import RF-Genesis: {e}")
        print("\nPlease ensure RF-Genesis is properly installed:")
        print(f"  cd {RF_GENESIS_DIR}")
        print("  pip install -r requirements.txt")
        print("  sh setup.sh")
        sys.exit(1)

    # Setup output directory
    if args.name is None:
        args.name = f"output_{int(time.time())}"

    output_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Prompt: {args.prompt}")
    print(f"  Output: {output_dir}")
    print("-" * 60)

    # Step 1: Generate body motion using MDM
    motion_file = os.path.join(output_dir, 'obj_diff.npz')

    if not os.path.exists(motion_file):
        print("\n[Step 1/4] Generating human body motion with MDM...")
        object_diff.generate(args.prompt, output_dir)
    else:
        print("\n[Step 1/4] Motion file exists, skipping...")

    # Step 2: Ray tracing for body PIRs
    print("\n[Step 2/4] Rendering body PIRs via ray tracing...")

    original_dir = os.getcwd()
    os.chdir(str(RF_GENESIS_DIR / "genesis"))

    try:
        relative_path = os.path.join("../", output_dir, 'obj_diff.npz')
        body_pir, body_aux = pathtracer.trace(relative_path)
    finally:
        os.chdir(original_dir)

    # Step 3: Environment PIR (optional)
    env_pir = None
    if not args.no_environment:
        print("\n[Step 3/4] Generating environment PIRs...")
        try:
            from genesis.environment_diffusion import environemnt_diff
            env_diff = environemnt_diff.EnvironmentDiffusion(lora_path="Asixa/RFLoRA")
            env_pir = env_diff.generate("")
        except Exception as e:
            print(f"  [Warning] Environment generation failed: {e}")
            print("  Continuing without environment PIR...")
    else:
        print("\n[Step 3/4] Skipping environment generation...")

    # Step 4: Generate radar signal frames
    print("\n[Step 4/4] Generating radar signal frames...")

    radar_config = str(RF_GENESIS_DIR / "models" / "TI1843_config.json")
    radar_frames = signal_generator.generate_signal_frames(
        body_pir, body_aux, env_pir,
        radar_config=radar_config
    )

    print(f"  Radar frames shape: {radar_frames.shape}")

    # Save radar frames
    radar_output = os.path.join(output_dir, 'radar_frames.npy')
    np.save(radar_output, radar_frames)
    print(f"  Saved to {radar_output}")

    # Visualization
    if not args.no_visualize:
        print("\n[Extra] Generating visualization video...")
        torch.set_default_device('cpu')

        visualize.save_video(
            radar_config,
            radar_output,
            motion_file,
            os.path.join(output_dir, 'output.mp4')
        )
        print("  Video saved!")

    print("\n" + "=" * 60)
    print("RF-Genesis Pipeline Complete!")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
