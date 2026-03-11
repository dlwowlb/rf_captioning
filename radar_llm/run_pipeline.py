#!/usr/bin/env python3
"""
RadarLLM End-to-End Pipeline
=============================

Complete pipeline:
  1. Generate training data (Text → HY-Motion → RF-Genesis → Point Clouds)
  2. Train Stage 1: Radar Tokenizer (Aggregate VQ-VAE)
  3. Train Stage 2: Radar-aware Language Model (FLAN-T5-small)

Usage:
    # Full pipeline (generate data + train both stages):
    python -m radar_llm.run_pipeline --mode all

    # Generate data only:
    python -m radar_llm.run_pipeline --mode generate --max-prompts 10

    # Train only (data already generated):
    python -m radar_llm.run_pipeline --mode train --data-dir ./data/generated

    # Stage 1 only:
    python -m radar_llm.run_pipeline --mode stage1 --data-dir ./data/generated

    # Stage 2 only:
    python -m radar_llm.run_pipeline --mode stage2 \
        --data-dir ./data/generated \
        --tokenizer-ckpt ./outputs/tokenizer/best_model.pt
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_data_generation(args):
    """Step 1: Generate training data from text prompts."""
    print("\n" + "=" * 70)
    print(" STEP 1: Generating Training Data")
    print("=" * 70)

    cmd = [
        sys.executable, "-m", "radar_llm.generate_data",
        "--output-dir", args.data_dir,
        "--duration", str(args.duration),
        "--num-points", str(args.num_points),
        "--num-augments", str(args.num_augments),
        "--train-ratio", str(args.train_ratio),
        "--device", args.device,
    ]
    if args.max_prompts:
        cmd += ["--max-prompts", str(args.max_prompts)]
    if args.model_path:
        cmd += ["--model-path", args.model_path]
    if args.prompts_file:
        cmd += ["--prompts-file", args.prompts_file]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print("Data generation failed!")
        sys.exit(1)
    print("Data generation complete!")


def run_stage1(args):
    """Step 2: Train Radar Tokenizer (VQ-VAE)."""
    print("\n" + "=" * 70)
    print(" STEP 2: Training Radar Tokenizer (Stage 1)")
    print("=" * 70)

    cmd = [
        sys.executable, "-m", "radar_llm.train_tokenizer",
        "--data_root", args.data_dir,
        "--output_dir", os.path.join(args.output_dir, "tokenizer"),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr_stage1),
        "--max_epochs", str(args.epochs_stage1),
        "--num_points", str(args.num_points),
        "--point_dim", str(args.point_dim),
        "--seq_len", str(args.seq_len),
        "--num_anchors", str(args.num_anchors),
        "--codebook_size", str(args.codebook_size),
        "--codebook_dim", str(args.codebook_dim),
        "--mask_ratio", str(args.mask_ratio),
        "--num_workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--device", args.device,
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print("Stage 1 training failed!")
        sys.exit(1)
    print("Stage 1 training complete!")


def run_stage2(args):
    """Step 3: Train Language Model."""
    print("\n" + "=" * 70)
    print(" STEP 3: Training Language Model (Stage 2)")
    print("=" * 70)

    tokenizer_ckpt = args.tokenizer_ckpt
    if tokenizer_ckpt is None:
        tokenizer_ckpt = os.path.join(args.output_dir, "tokenizer", "best_model.pt")

    if not os.path.exists(tokenizer_ckpt):
        print(f"Tokenizer checkpoint not found: {tokenizer_ckpt}")
        print("Please run Stage 1 first or provide --tokenizer-ckpt")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "radar_llm.train_lm",
        "--data_root", args.data_dir,
        "--tokenizer_ckpt", tokenizer_ckpt,
        "--output_dir", os.path.join(args.output_dir, "lm"),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr_stage2),
        "--max_epochs", str(args.epochs_stage2),
        "--num_points", str(args.num_points),
        "--point_dim", str(args.point_dim),
        "--seq_len", str(args.seq_len),
        "--num_anchors", str(args.num_anchors),
        "--codebook_size", str(args.codebook_size),
        "--codebook_dim", str(args.codebook_dim),
        "--embed_dim", str(args.embed_dim),
        "--text_model", args.text_model,
        "--num_workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--device", args.device,
    ]
    if args.use_continuous_features:
        cmd.append("--use_continuous_features")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print("Stage 2 training failed!")
        sys.exit(1)
    print("Stage 2 training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="RadarLLM End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "generate", "train", "stage1", "stage2"],
                        help="Pipeline mode: all, generate, train, stage1, stage2")

    # Data generation
    parser.add_argument("--data-dir", type=str, default="./data/generated",
                        help="Data directory (output for generation, input for training)")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--num-augments", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Limit prompts count (for quick testing)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="HY-Motion model path")
    parser.add_argument("--prompts-file", type=str, default=None,
                        help="Custom prompts JSON file")

    # Model architecture
    parser.add_argument("--num-points", type=int, default=64)
    parser.add_argument("--point-dim", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--num-anchors", type=int, default=16)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--codebook-dim", type=int, default=512)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--text-model", type=str, default="google/flan-t5-small")
    parser.add_argument("--use-continuous-features", action="store_true")

    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-stage1", type=float, default=1e-4)
    parser.add_argument("--lr-stage2", type=float, default=5e-5)
    parser.add_argument("--epochs-stage1", type=int, default=100)
    parser.add_argument("--epochs-stage2", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # Output
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--tokenizer-ckpt", type=str, default=None,
                        help="Pre-trained tokenizer checkpoint (for stage2 only)")

    args = parser.parse_args()

    print("=" * 70)
    print(" RadarLLM End-to-End Pipeline")
    print("=" * 70)
    print(f" Mode:       {args.mode}")
    print(f" Data dir:   {args.data_dir}")
    print(f" Output dir: {args.output_dir}")
    print(f" Device:     {args.device}")
    print("=" * 70)

    if args.mode == "all":
        run_data_generation(args)
        run_stage1(args)
        run_stage2(args)
    elif args.mode == "generate":
        run_data_generation(args)
    elif args.mode == "train":
        run_stage1(args)
        run_stage2(args)
    elif args.mode == "stage1":
        run_stage1(args)
    elif args.mode == "stage2":
        run_stage2(args)

    print("\n" + "=" * 70)
    print(" Pipeline Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
