"""
Stage 2: Train Radar-aware Language Model.

Uses the frozen tokenizer from Stage 1 to convert radar point clouds into
discrete tokens, then trains a FLAN-T5-small model to generate natural
language captions from those tokens.

Usage:
    # With real data:
    python -m radar_llm.train_lm \
        --data_root ./data \
        --tokenizer_ckpt outputs/tokenizer/best_model.pt \
        --batch_size 32

    # With dummy data (for testing):
    python -m radar_llm.train_lm --use_dummy --batch_size 8 --max_epochs 5

    # Resume:
    python -m radar_llm.train_lm --resume outputs/lm/checkpoint_epoch_10.pt
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
from radar_llm.models.radar_lm import RadarLanguageModel
from radar_llm.data.dataset import build_dataloader
from radar_llm.config import TokenizerConfig, LanguageModelConfig, TrainConfig


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Radar-aware Language Model (Stage 2)")

    # Data
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--use_dummy", action="store_true")
    parser.add_argument("--dummy_size", type=int, default=1000)

    # Tokenizer (frozen, from Stage 1)
    parser.add_argument("--tokenizer_ckpt", type=str, default=None,
                        help="Path to trained tokenizer checkpoint")
    parser.add_argument("--num_points", type=int, default=64)
    parser.add_argument("--point_dim", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--num_anchors", type=int, default=16)
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--codebook_dim", type=int, default=512)

    # Language model
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--text_model", type=str,
                        default="google/flan-t5-small")
    parser.add_argument("--max_text_len", type=int, default=128)
    parser.add_argument("--max_radar_tokens", type=int, default=16)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--use_continuous_features", action="store_true",
                        help="Use continuous radar features instead of discrete tokens")

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Logging & saving
    parser.add_argument("--output_dir", type=str, default="./outputs/lm")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--gen_every", type=int, default=5,
                        help="Generate sample captions every N epochs")

    # Resume
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def load_frozen_tokenizer(args, device):
    """Load and freeze the pretrained radar tokenizer."""
    tokenizer = RadarTokenizer(
        point_dim=args.point_dim,
        seq_len=args.seq_len,
        num_points=args.num_points,
        num_anchors=args.num_anchors,
        codebook_size=args.codebook_size,
        codebook_dim=args.codebook_dim,
    ).to(device)

    if args.tokenizer_ckpt and os.path.exists(args.tokenizer_ckpt):
        ckpt = torch.load(args.tokenizer_ckpt, map_location=device)
        tokenizer.load_state_dict(ckpt["model"])
        print(f"Loaded tokenizer from {args.tokenizer_ckpt}")
    else:
        print("WARNING: No tokenizer checkpoint provided. "
              "Using randomly initialized tokenizer.")

    # Freeze tokenizer
    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad = False

    return tokenizer


def tokenize_batch(tokenizer, lm, point_clouds, device, use_continuous):
    """Convert point clouds to radar tokens using frozen tokenizer."""
    with torch.no_grad():
        token_ids, z_q = tokenizer.encode(point_clouds)

    if use_continuous:
        return token_ids, z_q
    return token_ids, None


def prepare_text_batch(lm_model, captions, device, max_len=128):
    """Tokenize captions for T5 training."""
    tokenizer = lm_model.tokenizer

    # Encode captions
    encoding = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Labels: same as input_ids but with padding tokens replaced by -100
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return input_ids, attention_mask, labels


def train_one_epoch(lm_model, tokenizer, dataloader, optimizer, scheduler,
                    device, epoch, args):
    lm_model.train()
    total_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(dataloader):
        point_clouds = batch["point_clouds"].to(device)
        captions = batch["caption"]

        # Tokenize radar point clouds (frozen)
        radar_token_ids, radar_features = tokenize_batch(
            tokenizer, lm_model, point_clouds, device,
            args.use_continuous_features
        )

        # Prepare text targets
        text_input_ids, text_mask, labels = prepare_text_batch(
            lm_model, captions, device, args.max_text_len
        )

        # Forward through language model
        outputs = lm_model(
            radar_token_ids=radar_token_ids,
            radar_features=radar_features,
            text_input_ids=text_input_ids,
            text_attention_mask=text_mask,
            labels=labels,
        )

        loss = outputs["loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(lm_model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        if (step + 1) % args.log_every == 0:
            avg_loss = total_loss / num_batches
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [Epoch {epoch} Step {step+1}/{len(dataloader)}] "
                f"loss={avg_loss:.4f} lr={lr:.2e}"
            )

    return {"loss": total_loss / max(num_batches, 1)}


@torch.no_grad()
def evaluate(lm_model, tokenizer, dataloader, device, args):
    lm_model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        point_clouds = batch["point_clouds"].to(device)
        captions = batch["caption"]

        radar_token_ids, radar_features = tokenize_batch(
            tokenizer, lm_model, point_clouds, device,
            args.use_continuous_features
        )

        text_input_ids, text_mask, labels = prepare_text_batch(
            lm_model, captions, device, args.max_text_len
        )

        outputs = lm_model(
            radar_token_ids=radar_token_ids,
            radar_features=radar_features,
            text_input_ids=text_input_ids,
            text_attention_mask=text_mask,
            labels=labels,
        )

        total_loss += outputs["loss"].item()
        num_batches += 1

    return {"loss": total_loss / max(num_batches, 1)}


@torch.no_grad()
def generate_samples(lm_model, tokenizer, dataloader, device, args,
                     num_samples=5):
    """Generate sample captions for qualitative evaluation."""
    lm_model.eval()
    samples = []

    for batch in dataloader:
        point_clouds = batch["point_clouds"].to(device)
        captions = batch["caption"]
        sample_ids = batch["sample_id"]

        radar_token_ids, radar_features = tokenize_batch(
            tokenizer, lm_model, point_clouds, device,
            args.use_continuous_features
        )

        generated = lm_model.generate_caption(
            radar_token_ids=radar_token_ids,
            radar_features=radar_features,
            max_length=args.max_text_len,
        )

        for i in range(min(len(generated), num_samples - len(samples))):
            samples.append({
                "id": sample_ids[i],
                "ground_truth": captions[i],
                "generated": generated[i],
            })

        if len(samples) >= num_samples:
            break

    return samples


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load frozen tokenizer
    radar_tokenizer = load_frozen_tokenizer(args, device)

    # Build language model
    lm_model = RadarLanguageModel(
        radar_vocab_size=args.codebook_size,
        embed_dim=args.embed_dim,
        text_model_name=args.text_model,
        max_radar_tokens=args.max_radar_tokens,
        max_text_len=args.max_text_len,
        label_smoothing=args.label_smoothing,
        codebook_dim=args.codebook_dim,
    ).to(device)

    num_params = sum(p.numel() for p in lm_model.parameters()
                     if p.requires_grad)
    print(f"Language model trainable parameters: {num_params / 1e6:.2f}M")

    # Build dataloaders
    train_loader = build_dataloader(
        data_root=args.data_root, split="train",
        batch_size=args.batch_size, num_workers=args.num_workers,
        seq_len=args.seq_len, num_points=args.num_points,
        point_dim=args.point_dim, load_motion=False, load_captions=True,
        use_dummy=args.use_dummy, dummy_size=args.dummy_size,
    )
    val_loader = build_dataloader(
        data_root=args.data_root, split="val",
        batch_size=args.batch_size, num_workers=args.num_workers,
        seq_len=args.seq_len, num_points=args.num_points,
        point_dim=args.point_dim, load_motion=False, load_captions=True,
        use_dummy=args.use_dummy, dummy_size=max(args.dummy_size // 5, 100),
    )

    # Optimizer (only LM parameters, tokenizer is frozen)
    optimizer = AdamW(
        lm_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.max_epochs

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=args.warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(total_steps - args.warmup_steps, 1), eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_steps]
    )

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        lm_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

    # Training
    print(f"\n{'='*60}")
    print(f"Stage 2: Training Radar-aware Language Model")
    print(f"{'='*60}")
    print(f"  Backbone: {args.text_model}")
    print(f"  Epochs: {args.max_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Radar vocab: {args.codebook_size} tokens")
    print(f"  Continuous features: {args.use_continuous_features}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.max_epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            lm_model, radar_tokenizer, train_loader, optimizer, scheduler,
            device, epoch, args
        )

        epoch_time = time.time() - t0
        print(
            f"Epoch {epoch} ({epoch_time:.1f}s): "
            f"train_loss={train_metrics['loss']:.4f}"
        )

        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            val_metrics = evaluate(
                lm_model, radar_tokenizer, val_loader, device, args
            )
            print(f"  Val: loss={val_metrics['loss']:.4f}")

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save({
                    "model": lm_model.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                }, os.path.join(args.output_dir, "best_model.pt"))
                print("  -> Saved best model!")

        # Generate sample captions
        if (epoch + 1) % args.gen_every == 0:
            samples = generate_samples(
                lm_model, radar_tokenizer, val_loader, device, args
            )
            print("\n  --- Sample Captions ---")
            for s in samples:
                print(f"  [{s['id']}]")
                print(f"    GT:  {s['ground_truth']}")
                print(f"    Gen: {s['generated']}")
            print()

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(
                args.output_dir, f"checkpoint_epoch_{epoch+1}.pt"
            )
            torch.save({
                "model": lm_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            }, ckpt_path)
            print(f"  -> Saved checkpoint: {ckpt_path}")

    # Final save
    torch.save({
        "model": lm_model.state_dict(),
        "epoch": args.max_epochs - 1,
    }, os.path.join(args.output_dir, "final_model.pt"))
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
