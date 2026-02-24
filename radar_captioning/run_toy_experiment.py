#!/usr/bin/env python3
"""
Run Complete Toy Experiment: Dataset Generation → Stage 1 → Stage 2 → Validation.

This script runs the full radar captioning pipeline on a toy dataset to verify
that losses decrease and embeddings align properly.

Usage:
    python -m radar_captioning.run_toy_experiment [--num_samples 300] [--device cpu]
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .config import FullConfig
from .data.toy_dataset import generate_toy_dataset
from .train_stage1 import train_stage1
from .train_stage2 import train_stage2
from .validate_stage1 import validate_stage1
from .validate_stage2 import validate_stage2


def plot_training_history(output_dir: str):
    """Plot combined training loss curves for Stage 1 and Stage 2."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Stage 1
    s1_path = os.path.join(output_dir, "stage1", "stage1_history.json")
    if os.path.exists(s1_path):
        with open(s1_path) as f:
            s1 = json.load(f)
        epochs = range(1, len(s1["train_loss"]) + 1)
        axes[0].plot(epochs, s1["train_loss"], "b-", label="Train Loss (total)")
        axes[0].plot(epochs, s1["val_loss"], "r--", label="Val Loss")
        axes[0].plot(epochs, s1["train_mse"], "g:", label="Train MSE")
        axes[0].plot(epochs, s1["train_contrastive"], "m:", label="Train Contrastive")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Stage 1: Representation Alignment")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Show loss decrease
        first = s1["train_loss"][0]
        last = s1["train_loss"][-1]
        axes[0].annotate(
            f"Loss: {first:.3f} → {last:.3f}\n({(1-last/first)*100:.1f}% decrease)",
            xy=(len(epochs), last), fontsize=9,
            bbox=dict(boxstyle="round", fc="yellow", alpha=0.5),
        )

    # Stage 2
    s2_path = os.path.join(output_dir, "stage2", "stage2_history.json")
    if os.path.exists(s2_path):
        with open(s2_path) as f:
            s2 = json.load(f)
        epochs = range(1, len(s2["train_loss"]) + 1)
        axes[1].plot(epochs, s2["train_loss"], "b-", label="Train Loss")
        axes[1].plot(epochs, s2["val_loss"], "r--", label="Val Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Stage 2: Sentence Generation")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        first = s2["train_loss"][0]
        last = s2["train_loss"][-1]
        axes[1].annotate(
            f"Loss: {first:.3f} → {last:.3f}\n({(1-last/first)*100:.1f}% decrease)",
            xy=(len(epochs), last), fontsize=9,
            bbox=dict(boxstyle="round", fc="yellow", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"Training curves saved → {output_dir}/training_curves.png")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete toy experiment for radar captioning"
    )
    parser.add_argument("--num_samples", type=int, default=300,
                        help="Number of toy samples (100-500)")
    parser.add_argument("--num_frames", type=int, default=30,
                        help="Frames per sample")
    parser.add_argument("--stage1_epochs", type=int, default=50,
                        help="Stage 1 training epochs")
    parser.add_argument("--stage2_epochs", type=int, default=30,
                        help="Stage 2 training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu/cuda)")
    parser.add_argument("--output_dir", type=str, default="output/toy_experiment",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_stage2", action="store_true",
                        help="Skip Stage 2 (only run Stage 1)")
    args = parser.parse_args()

    # Config
    config = FullConfig()
    config.data.num_samples = args.num_samples
    config.data.num_frames = args.num_frames
    config.stage1.num_epochs = args.stage1_epochs
    config.stage1.batch_size = args.batch_size
    config.stage2.num_epochs = args.stage2_epochs
    config.stage2.batch_size = args.batch_size
    config.device = args.device
    config.seed = args.seed

    data_dir = os.path.join(args.output_dir, "data")
    output_dir = args.output_dir

    print("=" * 60)
    print("  Radar Captioning - Toy Experiment")
    print("=" * 60)
    print(f"  Samples: {args.num_samples}")
    print(f"  Frames: {args.num_frames}")
    print(f"  Device: {args.device}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    start_time = time.time()

    # ================================================================
    # Step 1: Generate Toy Dataset
    # ================================================================
    print("\n[1/5] Generating toy dataset...")
    generate_toy_dataset(
        num_samples=args.num_samples,
        num_frames=args.num_frames,
        heatmap_size=config.data.heatmap_size,
        output_dir=data_dir,
        seed=args.seed,
    )

    # ================================================================
    # Step 2: Train Stage 1 (Representation Alignment)
    # ================================================================
    print("\n[2/5] Training Stage 1: Representation Alignment...")
    s1_dir = os.path.join(output_dir, "stage1")
    radar_qformer, text_encoder, s1_history = train_stage1(
        data_dir=data_dir,
        output_dir=s1_dir,
        config=config,
    )

    # ================================================================
    # Step 3: Validate Stage 1
    # ================================================================
    print("\n[3/5] Validating Stage 1: Embedding Visualization + Cosine Similarity...")
    s1_val_dir = os.path.join(output_dir, "validation", "stage1")
    s1_results = validate_stage1(
        data_dir=data_dir,
        ckpt_path=os.path.join(s1_dir, "best_stage1.pt"),
        output_dir=s1_val_dir,
        config=config,
    )

    if not args.skip_stage2:
        # ================================================================
        # Step 4: Train Stage 2 (Sentence Generation)
        # ================================================================
        print("\n[4/5] Training Stage 2: Sentence Generation...")
        s2_dir = os.path.join(output_dir, "stage2")
        s2_history = train_stage2(
            data_dir=data_dir,
            stage1_ckpt=os.path.join(s1_dir, "best_stage1.pt"),
            output_dir=s2_dir,
            config=config,
        )

        # ================================================================
        # Step 5: Validate Stage 2
        # ================================================================
        print("\n[5/5] Validating Stage 2: JSON Slots + Sentence Reconstruction...")
        s2_val_dir = os.path.join(output_dir, "validation", "stage2")
        s2_results = validate_stage2(
            data_dir=data_dir,
            stage1_ckpt=os.path.join(s1_dir, "best_stage1.pt"),
            stage2_ckpt=os.path.join(s2_dir, "best_stage2.pt"),
            output_dir=s2_val_dir,
            config=config,
        )
    else:
        print("\n[4-5/5] Skipping Stage 2 (--skip_stage2)")
        s2_history = None
        s2_results = None

    # ================================================================
    # Plot combined training curves
    # ================================================================
    plot_training_history(output_dir)

    # ================================================================
    # Summary
    # ================================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  EXPERIMENT COMPLETE")
    print("=" * 60)

    print(f"\n  Stage 1 (Representation Alignment):")
    print(f"    Train loss: {s1_history['train_loss'][0]:.4f} → {s1_history['train_loss'][-1]:.4f}")
    print(f"    Val loss:   {s1_history['val_loss'][0]:.4f} → {s1_history['val_loss'][-1]:.4f}")
    print(f"    Top-1 Retrieval Accuracy: {s1_results['top1_accuracy']:.1f}%")
    print(f"    Mean Cosine Sim Gap: {s1_results['mean_sim_gap']:.4f}")

    if s2_history is not None:
        print(f"\n  Stage 2 (Sentence Generation):")
        print(f"    Train loss: {s2_history['train_loss'][0]:.4f} → {s2_history['train_loss'][-1]:.4f}")
        print(f"    Val loss:   {s2_history['val_loss'][0]:.4f} → {s2_history['val_loss'][-1]:.4f}")

    if s2_results is not None:
        print(f"    Doppler Sign Accuracy: {s2_results['doppler_accuracy']:.1f}%")
        print(f"    Range within 3m: {s2_results['range_accuracy_3m']:.1f}%")

    print(f"\n  Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
