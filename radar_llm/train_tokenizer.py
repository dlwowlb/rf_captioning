"""
Stage 1: Train Motion-guided Radar Tokenizer (Aggregate VQ-VAE).

Trains the radar tokenizer to convert point cloud sequences into discrete tokens.
Losses:
  - Reconstruction loss (MSE on point clouds)
  - VQ commitment loss (codebook learning)
  - Motion alignment loss (radar features ↔ motion features)

Usage:
    # With real data:
    python -m radar_llm.train_tokenizer --data_root ./data --batch_size 32

    # With dummy data (for testing):
    python -m radar_llm.train_tokenizer --use_dummy --batch_size 16 --max_epochs 5

    # Resume from checkpoint:
    python -m radar_llm.train_tokenizer --resume outputs/tokenizer/checkpoint_epoch_10.pt
"""

import os
import sys
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from radar_llm.models.tokenizer import RadarTokenizer
from radar_llm.data.dataset import build_dataloader
from radar_llm.config import TokenizerConfig, TrainConfig


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Radar Tokenizer (Stage 1)")

    # Data
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--use_dummy", action="store_true",
                        help="Use dummy dataset for testing")
    parser.add_argument("--dummy_size", type=int, default=1000)

    # Model
    parser.add_argument("--num_points", type=int, default=64)
    parser.add_argument("--point_dim", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--num_anchors", type=int, default=16)
    parser.add_argument("--encoder_dim", type=int, default=128)
    parser.add_argument("--encoder_out_dim", type=int, default=512)
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--codebook_dim", type=int, default=512)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--decoder_heads", type=int, default=8)
    parser.add_argument("--decoder_layers", type=int, default=4)

    # Loss weights
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--vq_weight", type=float, default=1.0)
    parser.add_argument("--motion_align_weight", type=float, default=0.5)
    parser.add_argument("--commitment_cost", type=float, default=0.25)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Logging & saving
    parser.add_argument("--output_dir", type=str, default="./outputs/tokenizer")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=5)

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch,
                    args):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_vq = 0.0
    total_motion = 0.0
    total_perplexity = 0.0
    num_batches = 0

    for step, batch in enumerate(dataloader):
        point_clouds = batch["point_clouds"].to(device)
        motion_seq = batch.get("motion_seq")
        if motion_seq is not None:
            motion_seq = motion_seq.to(device)

        # Forward
        outputs = model(point_clouds, motion_seq=motion_seq)

        loss = outputs["loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_recon += outputs["recon_loss"].item()
        total_vq += outputs["vq_loss"].item()
        total_motion += outputs["motion_loss"].item()
        total_perplexity += outputs["perplexity"].item()
        num_batches += 1

        if (step + 1) % args.log_every == 0:
            avg_loss = total_loss / num_batches
            avg_recon = total_recon / num_batches
            avg_vq = total_vq / num_batches
            avg_motion = total_motion / num_batches
            avg_ppl = total_perplexity / num_batches
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [Epoch {epoch} Step {step+1}/{len(dataloader)}] "
                f"loss={avg_loss:.4f} recon={avg_recon:.4f} "
                f"vq={avg_vq:.4f} motion={avg_motion:.4f} "
                f"ppl={avg_ppl:.1f} lr={lr:.2e}"
            )

    return {
        "loss": total_loss / max(num_batches, 1),
        "recon_loss": total_recon / max(num_batches, 1),
        "vq_loss": total_vq / max(num_batches, 1),
        "motion_loss": total_motion / max(num_batches, 1),
        "perplexity": total_perplexity / max(num_batches, 1),
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_vq = 0.0
    total_perplexity = 0.0
    num_batches = 0

    for batch in dataloader:
        point_clouds = batch["point_clouds"].to(device)
        motion_seq = batch.get("motion_seq")
        if motion_seq is not None:
            motion_seq = motion_seq.to(device)

        outputs = model(point_clouds, motion_seq=motion_seq)

        total_loss += outputs["loss"].item()
        total_recon += outputs["recon_loss"].item()
        total_vq += outputs["vq_loss"].item()
        total_perplexity += outputs["perplexity"].item()
        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "recon_loss": total_recon / max(num_batches, 1),
        "vq_loss": total_vq / max(num_batches, 1),
        "perplexity": total_perplexity / max(num_batches, 1),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build model
    model = RadarTokenizer(
        point_dim=args.point_dim,
        seq_len=args.seq_len,
        num_points=args.num_points,
        num_anchors=args.num_anchors,
        encoder_dim=args.encoder_dim,
        encoder_out_dim=args.encoder_out_dim,
        codebook_size=args.codebook_size,
        codebook_dim=args.codebook_dim,
        commitment_cost=args.commitment_cost,
        mask_ratio=args.mask_ratio,
        decoder_heads=args.decoder_heads,
        decoder_layers=args.decoder_layers,
        recon_weight=args.recon_weight,
        vq_weight=args.vq_weight,
        motion_align_weight=args.motion_align_weight,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # Build dataloaders
    train_loader = build_dataloader(
        data_root=args.data_root, split="train",
        batch_size=args.batch_size, num_workers=args.num_workers,
        seq_len=args.seq_len, num_points=args.num_points,
        point_dim=args.point_dim, load_motion=True, load_captions=False,
        use_dummy=args.use_dummy, dummy_size=args.dummy_size,
    )
    val_loader = build_dataloader(
        data_root=args.data_root, split="val",
        batch_size=args.batch_size, num_workers=args.num_workers,
        seq_len=args.seq_len, num_points=args.num_points,
        point_dim=args.point_dim, load_motion=True, load_captions=False,
        use_dummy=args.use_dummy, dummy_size=max(args.dummy_size // 5, 100),
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.max_epochs

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=args.warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - args.warmup_steps, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_steps]
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

    # Training loop
    print(f"\n{'='*60}")
    print(f"Stage 1: Training Radar Tokenizer (Aggregate VQ-VAE)")
    print(f"{'='*60}")
    print(f"  Epochs: {args.max_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Codebook: {args.codebook_size} x {args.codebook_dim}")
    print(f"  Mask ratio: {args.mask_ratio}")
    print(f"  Anchors: {args.num_anchors}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.max_epochs):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args
        )

        epoch_time = time.time() - t0

        print(
            f"Epoch {epoch} ({epoch_time:.1f}s): "
            f"train_loss={train_metrics['loss']:.4f} "
            f"recon={train_metrics['recon_loss']:.4f} "
            f"vq={train_metrics['vq_loss']:.4f} "
            f"motion={train_metrics['motion_loss']:.4f} "
            f"ppl={train_metrics['perplexity']:.1f}"
        )

        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            val_metrics = evaluate(model, val_loader, device)
            print(
                f"  Val: loss={val_metrics['loss']:.4f} "
                f"recon={val_metrics['recon_loss']:.4f} "
                f"ppl={val_metrics['perplexity']:.1f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                }, os.path.join(args.output_dir, "best_model.pt"))
                print("  -> Saved best model!")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(
                args.output_dir, f"checkpoint_epoch_{epoch+1}.pt"
            )
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            }, ckpt_path)
            print(f"  -> Saved checkpoint: {ckpt_path}")

    # Save final model
    torch.save({
        "model": model.state_dict(),
        "epoch": args.max_epochs - 1,
    }, os.path.join(args.output_dir, "final_model.pt"))
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
