#!/usr/bin/env python3
"""End-to-end pipeline: text prompt -> HY-Motion -> RF-Genesis -> Doppler analysis.

Usage:
    python examples/motion_to_doppler.py \
        --prompt "a person walking forward" \
        --duration 3.0 \
        --output-dir output/test

    # Skip motion generation if obj_diff.npz already exists:
    python examples/motion_to_doppler.py \
        --prompt "a person walking forward" \
        --duration 3.0 \
        --output-dir output/test \
        --skip-motion

    # Skip environment PIR generation (faster):
    python examples/motion_to_doppler.py \
        --prompt "a person walking forward" \
        --duration 3.0 \
        --output-dir output/test \
        --no-environment
"""
import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup: add project roots so imports work from the examples/ directory
# ---------------------------------------------------------------------------
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ_ROOT, "HY-Motion-1.0"))
sys.path.insert(0, os.path.join(_PROJ_ROOT, "RF-Genesis"))


# ---------------------------------------------------------------------------
# Doppler spectrogram helpers
# ---------------------------------------------------------------------------
def compute_doppler_spectrogram(radar_frames, radar_cfg):
    """Compute Doppler spectrogram: (num_frames, num_doppler_bins).

    For each radar frame, compute Range-Doppler map via rangeFFT + dopplerFFT,
    then take the max power across all range bins for each Doppler bin.

    Args:
        radar_frames: (N, num_tx, num_rx, num_chirps, num_adc) complex array
        radar_cfg: Radar config object or path to JSON config

    Returns:
        spectrogram: (N, num_doppler_bins) real array in dB
        doppler_axis: (num_doppler_bins,) velocity axis in m/s
        time_axis: (N,) time axis in seconds
    """
    from genesis.visualization.pointcloud import (
        FrameConfig, rangeFFT, dopplerFFT, clutter_removal,
    )
    from genesis.raytracing.radar import Radar

    if isinstance(radar_cfg, str):
        radar = Radar(radar_cfg)
    else:
        radar = radar_cfg

    frame_cfg = FrameConfig(radar)
    num_frames = radar_frames.shape[0]
    num_doppler_bins = frame_cfg.numDopplerBins

    spectrogram = np.zeros((num_frames, num_doppler_bins), dtype=np.float64)

    for i in range(num_frames):
        frame = radar_frames[i]
        range_result = rangeFFT(frame, frame_cfg)
        range_result = clutter_removal(range_result, axis=2)
        doppler_result = dopplerFFT(range_result, frame_cfg)

        # Sum over all TX/RX antennas -> (num_doppler_bins, num_range_bins)
        doppler_sum = np.sum(doppler_result, axis=(0, 1))
        doppler_power = np.abs(doppler_sum)

        # Max across range bins for each Doppler bin -> (num_doppler_bins,)
        doppler_profile = np.max(doppler_power, axis=1)
        spectrogram[i] = doppler_profile

    # Convert to dB
    spectrogram_db = 20 * np.log10(spectrogram + 1e-12)

    # Doppler velocity axis
    doppler_res = frame_cfg.doppler_resolution
    doppler_axis = (np.arange(num_doppler_bins) - num_doppler_bins // 2) * doppler_res

    # Time axis
    time_axis = np.arange(num_frames) / radar.frame_per_second

    return spectrogram_db, doppler_axis, time_axis


def plot_range_doppler(radar_frames, radar_cfg, frame_idx=0):
    """Plot Range-Doppler map for a single frame.

    Returns:
        fig: matplotlib Figure
    """
    from genesis.visualization.pointcloud import (
        FrameConfig, rangeFFT, dopplerFFT, clutter_removal,
    )
    from genesis.raytracing.radar import Radar

    if isinstance(radar_cfg, str):
        radar = Radar(radar_cfg)
    else:
        radar = radar_cfg

    frame_cfg = FrameConfig(radar)
    frame = radar_frames[frame_idx]
    range_result = rangeFFT(frame, frame_cfg)
    range_result = clutter_removal(range_result, axis=2)
    doppler_result = dopplerFFT(range_result, frame_cfg)

    doppler_sum = np.sum(doppler_result, axis=(0, 1))
    doppler_db = 20 * np.log10(np.abs(doppler_sum) + 1e-12)

    range_res = frame_cfg.range_resolution
    doppler_res = frame_cfg.doppler_resolution
    num_range = frame_cfg.numRangeBins
    num_doppler = frame_cfg.numDopplerBins

    range_axis = np.arange(num_range) * range_res
    doppler_axis = (np.arange(num_doppler) - num_doppler // 2) * doppler_res

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        doppler_db,
        aspect="auto",
        origin="lower",
        extent=[range_axis[0], range_axis[-1], doppler_axis[0], doppler_axis[-1]],
        cmap="jet",
    )
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(f"Range-Doppler Map (frame {frame_idx})")
    fig.colorbar(im, ax=ax, label="Power (dB)")
    return fig


def plot_doppler_spectrogram(spectrogram_db, doppler_axis, time_axis):
    """Plot Doppler spectrogram (time vs velocity).

    Returns:
        fig: matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        spectrogram_db.T,
        aspect="auto",
        origin="lower",
        extent=[time_axis[0], time_axis[-1], doppler_axis[0], doppler_axis[-1]],
        cmap="jet",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Doppler Spectrogram")
    fig.colorbar(im, ax=ax, label="Power (dB)")
    return fig


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_motion_to_doppler(
    prompt,
    duration,
    output_dir,
    model_path=None,
    cfg_scale=5.0,
    seed=42,
    skip_motion=False,
    skip_environment=True,
    radar_config=None,
    disable_prompt_engineering=True,
):
    """Run the full text-to-Doppler pipeline.

    Steps:
      1. Generate motion with HY-Motion (or reuse existing obj_diff.npz)
      2. Run RF-Genesis ray tracing + radar signal generation
      3. Compute and save Range-Doppler map and Doppler spectrogram
    """
    import torch

    os.makedirs(output_dir, exist_ok=True)
    obj_diff_path = os.path.join(output_dir, "obj_diff.npz")

    # ------------------------------------------------------------------
    # Step 1: Generate motion with HY-Motion
    # ------------------------------------------------------------------
    if not skip_motion and not os.path.exists(obj_diff_path):
        print("[Pipeline] Step 1/3: Generating motion with HY-Motion...")
        from hymotion.utils.t2m_runtime import T2MRuntime

        if model_path is None:
            # Default model path
            model_path = os.path.join(_PROJ_ROOT, "HY-Motion-1.0", "ckpts", "tencent", "HY-Motion-1.0")

        config_path = os.path.join(model_path, "config.yml")
        ckpt_path = os.path.join(model_path, "latest.ckpt")
        skip_model = not os.path.exists(ckpt_path)

        runtime = T2MRuntime(
            config_path=config_path,
            ckpt_name=ckpt_path,
            skip_model_loading=skip_model,
            disable_prompt_engineering=disable_prompt_engineering,
        )

        seeds_csv = str(seed)
        _, _, model_output = runtime.generate_motion(
            text=prompt,
            seeds_csv=seeds_csv,
            duration=duration,
            cfg_scale=cfg_scale,
            output_format="dict",
            output_dir=output_dir,
        )

        # Convert HY-Motion output to RF-Genesis format via object_diff
        hymotion_output = {
            "rot6d": model_output["rot6d"],
            "transl": model_output["transl"],
            "keypoints3d": model_output.get("keypoints3d", None),
        }

        from genesis.object_diffusion import object_diff
        object_diff.generate(prompt, output_dir, hymotion_output=hymotion_output)
    else:
        if os.path.exists(obj_diff_path):
            print(f"[Pipeline] Step 1/3: Reusing existing {obj_diff_path}")
        else:
            print(f"[Pipeline] Step 1/3: Skipped (--skip-motion). Expecting {obj_diff_path} to exist.")
            if not os.path.exists(obj_diff_path):
                print(f"ERROR: {obj_diff_path} not found. Run without --skip-motion first.")
                sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: RF-Genesis ray tracing + radar signal generation
    # ------------------------------------------------------------------
    radar_frames_path = os.path.join(output_dir, "radar_frames.npy")

    if not os.path.exists(radar_frames_path):
        print("[Pipeline] Step 2/3: Running RF-Genesis (ray tracing + radar signal)...")

        # RF-Genesis run.py expects to be run from the RF-Genesis directory
        rfgen_dir = os.path.join(_PROJ_ROOT, "RF-Genesis")
        orig_dir = os.getcwd()
        os.chdir(rfgen_dir)

        try:
            from genesis.raytracing import pathtracer, signal_generator

            # Resolve radar config
            if radar_config is None:
                radar_config = os.path.join(rfgen_dir, "models", "TI1843_config.json")

            # Relative path for pathtracer (it operates from genesis/ subdir)
            os.chdir("genesis/")
            rel_obj_diff = os.path.relpath(obj_diff_path, os.getcwd())
            body_pir, body_aux = pathtracer.trace(rel_obj_diff)
            os.chdir(rfgen_dir)

            # Environment PIR
            env_pir = None
            if not skip_environment:
                from genesis.environment_diffusion import environemnt_diff
                envir_diff = environemnt_diff.EnvironmentDiffusion(lora_path="Asixa/RFLoRA")
                env_pir = envir_diff.generate(prompt)

            # Generate radar frames
            import torch
            radar_frames = signal_generator.generate_signal_frames(
                body_pir, body_aux, env_pir, radar_config=radar_config,
            )
            np.save(radar_frames_path, radar_frames)
            print(f"[Pipeline] Saved radar frames: {radar_frames.shape}")
        finally:
            os.chdir(orig_dir)
    else:
        print(f"[Pipeline] Step 2/3: Reusing existing {radar_frames_path}")

    # ------------------------------------------------------------------
    # Step 3: Doppler analysis and visualization
    # ------------------------------------------------------------------
    print("[Pipeline] Step 3/3: Computing Doppler analysis...")

    import torch
    torch.set_default_device("cpu")

    radar_frames = np.load(radar_frames_path)

    # Resolve radar config for analysis
    rfgen_dir = os.path.join(_PROJ_ROOT, "RF-Genesis")
    if radar_config is None:
        radar_config = os.path.join(rfgen_dir, "models", "TI1843_config.json")

    # Range-Doppler map (middle frame)
    mid_frame = len(radar_frames) // 2
    fig_rd = plot_range_doppler(radar_frames, radar_config, frame_idx=mid_frame)
    rd_path = os.path.join(output_dir, "range_doppler.png")
    fig_rd.savefig(rd_path, dpi=150, bbox_inches="tight")
    plt.close(fig_rd)
    print(f"  Saved Range-Doppler map: {rd_path}")

    # Doppler spectrogram (time vs velocity)
    spectrogram_db, doppler_axis, time_axis = compute_doppler_spectrogram(
        radar_frames, radar_config,
    )
    fig_ds = plot_doppler_spectrogram(spectrogram_db, doppler_axis, time_axis)
    ds_path = os.path.join(output_dir, "doppler_spectrogram.png")
    fig_ds.savefig(ds_path, dpi=150, bbox_inches="tight")
    plt.close(fig_ds)
    print(f"  Saved Doppler spectrogram: {ds_path}")

    # Save raw spectrogram data
    spec_data_path = os.path.join(output_dir, "doppler_spectrogram.npz")
    np.savez(
        spec_data_path,
        spectrogram_db=spectrogram_db,
        doppler_axis=doppler_axis,
        time_axis=time_axis,
    )
    print(f"  Saved spectrogram data: {spec_data_path}")

    print(f"\n[Pipeline] Done! Results saved in {output_dir}/")
    print(f"  - range_doppler.png        : Range-Doppler map (frame {mid_frame})")
    print(f"  - doppler_spectrogram.png  : Doppler spectrogram (time vs velocity)")
    print(f"  - doppler_spectrogram.npz  : Raw spectrogram data")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end: text prompt -> HY-Motion -> RF-Genesis -> Doppler analysis",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for motion generation")
    parser.add_argument("--duration", type=float, default=3.0, help="Motion duration in seconds")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--model-path", type=str, default=None, help="HY-Motion model checkpoint path")
    parser.add_argument("--radar-config", type=str, default=None, help="Radar JSON config path")
    parser.add_argument("--cfg-scale", type=float, default=5.0, help="CFG scale for motion generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-motion", action="store_true", help="Skip motion generation (reuse existing obj_diff.npz)")
    parser.add_argument("--no-environment", action="store_true", default=True, help="Skip environment PIR (default: skip)")
    parser.add_argument("--with-environment", action="store_true", help="Enable environment PIR generation")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join("output", f"doppler_{int(time.time())}")

    skip_env = args.no_environment and not args.with_environment

    run_motion_to_doppler(
        prompt=args.prompt,
        duration=args.duration,
        output_dir=args.output_dir,
        model_path=args.model_path,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        skip_motion=args.skip_motion,
        skip_environment=skip_env,
        radar_config=args.radar_config,
    )


if __name__ == "__main__":
    main()
