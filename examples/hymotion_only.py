#!/usr/bin/env python3
"""
HY-Motion Standalone Example
=============================

This example demonstrates how to use HY-Motion directly
to generate motion from text prompts.

Usage:
    python examples/hymotion_only.py \
        --prompt "a person walking forward" \
        --duration 3.0 \
        --output-dir output/motion_test

Requirements:
    - HY-Motion-1.0 model weights in HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# ============================================================================
# Path Setup
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
HY_MOTION_DIR = PROJECT_ROOT / "HY-Motion-1.0"

if HY_MOTION_DIR.exists():
    sys.path.insert(0, str(HY_MOTION_DIR))


def visualize_motion(smpl_data: np.ndarray, output_path: str, prompt: str):
    """Create simple visualization of motion trajectory."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    num_frames = smpl_data.shape[0]

    # Extract translation (first 3 values)
    if smpl_data.shape[1] >= 3:
        translation = smpl_data[:, :3]
    else:
        translation = np.zeros((num_frames, 3))

    # Create figure
    fig = plt.figure(figsize=(15, 5))

    # 1. 3D trajectory
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(translation[:, 0], translation[:, 1], translation[:, 2], 'b-', linewidth=2)
    ax1.scatter(translation[0, 0], translation[0, 1], translation[0, 2],
                c='green', s=100, marker='o', label='Start')
    ax1.scatter(translation[-1, 0], translation[-1, 1], translation[-1, 2],
                c='red', s=100, marker='x', label='End')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Trajectory')
    ax1.legend()

    # 2. Position over time
    ax2 = fig.add_subplot(132)
    time = np.arange(num_frames) / 30.0  # Assuming 30 FPS
    ax2.plot(time, translation[:, 0], label='X')
    ax2.plot(time, translation[:, 1], label='Y')
    ax2.plot(time, translation[:, 2], label='Z')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position')
    ax2.set_title('Position over Time')
    ax2.legend()
    ax2.grid(True)

    # 3. Motion parameter heatmap
    ax3 = fig.add_subplot(133)
    # Show subset of motion parameters
    display_params = min(50, smpl_data.shape[1])
    im = ax3.imshow(smpl_data[:, :display_params].T, aspect='auto', cmap='coolwarm')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Parameter')
    ax3.set_title(f'Motion Parameters (first {display_params})')
    plt.colorbar(im, ax=ax3)

    plt.suptitle(f"Motion: {prompt}\n({num_frames} frames)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[Viz] Saved motion visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="HY-Motion standalone example")
    parser.add_argument(
        "-p", "--prompt",
        required=True,
        help="Text prompt for motion generation"
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=3.0,
        help="Duration in seconds (default: 3.0)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Output directory"
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to HY-Motion model"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=5.0,
        help="CFG scale"
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of different seeds to generate"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("HY-Motion Standalone Pipeline")
    print("=" * 60)

    try:
        from hymotion.utils.t2m_runtime import T2MRuntime
        print("[OK] HY-Motion module loaded")
    except ImportError as e:
        print(f"[ERROR] Failed to import HY-Motion: {e}")
        print("\nPlease ensure HY-Motion is properly installed:")
        print(f"  cd {HY_MOTION_DIR}")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    # Setup paths
    if args.model_path is None:
        args.model_path = str(HY_MOTION_DIR / "ckpts" / "tencent" / "HY-Motion-1.0")

    config_path = os.path.join(args.model_path, "config.yml")
    ckpt_path = os.path.join(args.model_path, "latest.ckpt")

    if not os.path.exists(config_path):
        print(f"[ERROR] Config not found: {config_path}")
        print("\nPlease download the model:")
        print("  huggingface-cli download tencent/HY-Motion-1.0 \\")
        print(f"    --include 'HY-Motion-1.0/*' --local-dir {HY_MOTION_DIR}/ckpts/tencent")
        sys.exit(1)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"motion_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Prompt: {args.prompt}")
    print(f"  Duration: {args.duration}s")
    print(f"  Model: {args.model_path}")
    print(f"  Device: {args.device}")
    print(f"  Output: {output_dir}")
    print("-" * 60)

    # Initialize runtime
    print("\n[Step 1/3] Loading HY-Motion model...")

    runtime = T2MRuntime(
        config_path=config_path,
        ckpt_name=ckpt_path,
        device_ids=[0] if args.device == "cuda" and torch.cuda.is_available() else None,
        disable_prompt_engineering=True,
    )

    # Generate seeds
    seeds = [args.seed + i for i in range(args.num_seeds)]
    seeds_csv = ",".join(map(str, seeds))

    # Generate motion
    print(f"\n[Step 2/3] Generating motion for: '{args.prompt}'")

    html_content, fbx_files, model_output = runtime.generate_motion(
        text=args.prompt,
        seeds_csv=seeds_csv,
        duration=args.duration,
        cfg_scale=args.cfg_scale,
        output_format="dict",
        output_dir=output_dir,
    )

    print("[OK] Motion generated!")

    # Save motion data
    print("\n[Step 3/3] Saving motion data...")

    # Extract and save motion parameters
    if isinstance(model_output, dict):
        # Find the motion data in the output
        motion_data = None
        for key in ['motion', 'smpl_params', 'output', 'pred']:
            if key in model_output:
                motion_data = model_output[key]
                break

        if motion_data is None:
            # Try to find any tensor/array
            for key, value in model_output.items():
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    motion_data = value
                    print(f"  Found motion data in key: {key}")
                    break
    else:
        motion_data = model_output

    if motion_data is not None:
        if isinstance(motion_data, torch.Tensor):
            motion_data = motion_data.cpu().numpy()

        if motion_data.ndim == 3:
            motion_data = motion_data[0]

        # Save as numpy
        motion_path = os.path.join(output_dir, "motion_data.npy")
        np.save(motion_path, motion_data)
        print(f"  Saved: {motion_path}")
        print(f"  Shape: {motion_data.shape}")

        # Create visualization
        viz_path = os.path.join(output_dir, "motion_visualization.png")
        visualize_motion(motion_data, viz_path, args.prompt)
    else:
        print("  [Warning] Could not extract motion data from output")

    # Save metadata
    import json
    metadata = {
        'prompt': args.prompt,
        'duration': args.duration,
        'seeds': seeds,
        'cfg_scale': args.cfg_scale,
        'model_path': args.model_path,
        'timestamp': timestamp,
    }
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("HY-Motion Pipeline Complete!")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print("\nFiles generated:")
    for f in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  - {f} ({size / 1024:.1f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
