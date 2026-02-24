"""
Stage 1 Training: Representation Alignment.

Trains the Radar Q-Former to align radar heatmap embeddings (Z_Frame-Query)
with text embeddings (Z_text) from the frozen teacher text encoder.

Loss: MSE + Contrastive (InfoNCE)

Usage:
    python -m radar_captioning.train_stage1 \
        --data_dir data/toy_dataset \
        --output_dir output/stage1 \
        --num_epochs 50 \
        --batch_size 16
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import FullConfig
from .data.toy_dataset import RadarCaptionDataset, generate_toy_dataset
from .losses.alignment_loss import AlignmentLoss
from .models.radar_q_former import RadarQFormer
from .models.text_encoder import SurrogateTextEncoder


def collate_fn(batch):
    """Custom collate for variable-length text."""
    rd = torch.stack([b["rd"] for b in batch])
    ra = torch.stack([b["ra"] for b in batch])
    da = torch.stack([b["da"] for b in batch])
    texts = [b["text"] for b in batch]
    labels = [b["action_label"] for b in batch]
    return {"rd": rd, "ra": ra, "da": da, "texts": texts, "labels": labels}


def _pretrain_text_encoder(text_encoder, train_loader, device, epochs=10):
    """
    Pre-train the surrogate text encoder to produce diverse,
    action-discriminative embeddings before freezing.

    Uses a classification objective: predict action label from text embedding.
    This ensures different actions map to different regions in embedding space.
    """
    import torch.nn.functional as F

    text_encoder.train()

    # Collect unique action labels
    label_set = set()
    for batch in train_loader:
        label_set.update(batch["labels"])
    label_to_id = {l: i for i, l in enumerate(sorted(label_set))}
    num_classes = len(label_to_id)

    # Simple classification head (discarded after pretraining)
    classifier = nn.Linear(text_encoder.embed_dim, num_classes).to(device)

    optimizer = torch.optim.Adam(
        list(text_encoder.parameters()) + list(classifier.parameters()), lr=5e-4
    )

    for epoch in range(epochs):
        total_loss = 0
        n = 0
        for batch in train_loader:
            texts = batch["texts"]
            labels = batch["labels"]
            label_ids = torch.tensor([label_to_id[l] for l in labels], device=device)

            z = text_encoder(texts)  # (B, D)
            logits = classifier(z)   # (B, num_classes)
            loss = F.cross_entropy(logits, label_ids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1

        if (epoch + 1) % 5 == 0:
            print(f"  [TextEncoder pretrain] Epoch {epoch+1}/{epochs}, "
                  f"Loss: {total_loss/max(n,1):.4f}")

    # Verify diversity
    text_encoder.eval()
    with torch.no_grad():
        sample_texts = []
        sample_labels = []
        for batch in train_loader:
            sample_texts.extend(batch["texts"])
            sample_labels.extend(batch["labels"])
            if len(sample_texts) >= 20:
                break
        z = text_encoder(sample_texts[:20])
        z_norm = F.normalize(z, dim=-1)
        sim = torch.mm(z_norm, z_norm.t())
        off_diag = sim[~torch.eye(sim.shape[0], dtype=bool, device=device)]
        print(f"  [TextEncoder pretrain] Done. "
              f"Mean off-diag cosine sim: {off_diag.mean():.4f} "
              f"(lower = more diverse)")

    del classifier  # discard classification head


def train_stage1(
    data_dir: str,
    output_dir: str,
    config: FullConfig = None,
    curriculum_phase: str = None,
):
    """
    Train Stage 1: Representation Alignment.

    Args:
        data_dir: path to toy dataset
        output_dir: path to save checkpoints and logs
        config: full configuration
        curriculum_phase: 'basic', 'composition', or None for all
    """
    if config is None:
        config = FullConfig()

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(config.device)
    torch.manual_seed(config.seed)

    print("=" * 60)
    print("Stage 1: Representation Alignment Training")
    print("=" * 60)

    # Generate dataset if not exists
    if not os.path.exists(os.path.join(data_dir, "metadata.json")):
        print(f"Generating toy dataset at {data_dir}...")
        generate_toy_dataset(
            num_samples=config.data.num_samples,
            num_frames=config.data.num_frames,
            heatmap_size=config.data.heatmap_size,
            output_dir=data_dir,
            seed=config.seed,
        )

    # Load datasets
    train_dataset = RadarCaptionDataset(
        data_dir, split="train", curriculum_phase=curriculum_phase,
        num_frames=config.data.num_frames,
    )
    val_dataset = RadarCaptionDataset(
        data_dir, split="val", curriculum_phase=curriculum_phase,
        num_frames=config.data.num_frames,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.stage1.batch_size,
        shuffle=True, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.stage1.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Models
    radar_qformer = RadarQFormer(
        img_size=config.vit.img_size,
        patch_size=config.vit.patch_size,
        vit_embed_dim=config.vit.embed_dim,
        vit_depth=config.vit.depth,
        vit_num_heads=config.vit.num_heads,
        mask_ratio=config.tokenizer.mask_ratio,
        qformer_hidden_dim=config.qformer.hidden_dim,
        num_spatial_queries=config.qformer.num_spatial_queries,
        num_frame_queries=config.qformer.num_frame_queries,
        qformer_num_heads=config.qformer.num_heads,
        num_layers_spatial=config.qformer.num_layers_spatial,
        num_layers_temporal=config.qformer.num_layers_temporal,
        dropout=config.qformer.dropout,
    ).to(device)

    # Teacher text encoder (frozen)
    # In the real pipeline this is HY-Motion's text encoder.
    # SurrogateTextEncoder uses deterministic hash + keyword features,
    # already frozen by design.
    text_encoder = SurrogateTextEncoder(
        embed_dim=config.stage1.z_text_dim,
    ).to(device)
    text_encoder.eval()

    # Verify text encoder diversity
    with torch.no_grad():
        sample_texts = []
        for batch in train_loader:
            sample_texts.extend(batch["texts"])
            if len(sample_texts) >= 16:
                break
        z_sample = text_encoder(sample_texts[:16])
        sim = torch.mm(z_sample, z_sample.t())
        off_diag = sim[~torch.eye(16, dtype=bool, device=device)]
        print(f"Text encoder diversity: mean off-diag cosine sim = {off_diag.mean():.4f}")

    # Loss (contrastive-heavy since teacher is frozen)
    criterion = AlignmentLoss(
        mse_weight=config.stage1.mse_weight,
        contrastive_weight=config.stage1.contrastive_weight,
        temperature=config.stage1.temperature,
    )

    # Optimizer: only Q-Former parameters (text encoder is frozen)
    optimizer = torch.optim.AdamW(
        radar_qformer.parameters(),
        lr=config.stage1.lr,
        weight_decay=config.stage1.weight_decay,
    )

    # LR scheduler with warmup
    total_steps = config.stage1.num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.stage1.lr,
        total_steps=total_steps, pct_start=0.1,
    )

    # Training loop
    history = {"train_loss": [], "val_loss": [], "train_mse": [], "train_contrastive": []}
    best_val_loss = float("inf")

    for epoch in range(config.stage1.num_epochs):
        # Train
        radar_qformer.train()
        text_encoder.train()
        epoch_losses = {"total": 0, "mse": 0, "contrastive": 0}
        n_batches = 0

        for batch in train_loader:
            rd = batch["rd"].to(device)   # (B, T, H, W)
            ra = batch["ra"].to(device)
            da = batch["da"].to(device)
            texts = batch["texts"]

            # Forward
            frame_queries = radar_qformer(rd, ra, da)  # (B, Nq, D)
            z_radar = frame_queries.mean(dim=1)         # (B, D)
            z_text = text_encoder(texts)                 # (B, D)

            loss, loss_dict = criterion(z_radar, z_text)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(radar_qformer.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses["total"] += loss_dict["loss_total"]
            epoch_losses["mse"] += loss_dict["loss_mse"]
            epoch_losses["contrastive"] += loss_dict["loss_contrastive"]
            n_batches += 1

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        history["train_loss"].append(epoch_losses["total"])
        history["train_mse"].append(epoch_losses["mse"])
        history["train_contrastive"].append(epoch_losses["contrastive"])

        # Validation
        radar_qformer.eval()
        text_encoder.eval()
        val_loss_total = 0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                rd = batch["rd"].to(device)
                ra = batch["ra"].to(device)
                da = batch["da"].to(device)
                texts = batch["texts"]

                frame_queries = radar_qformer(rd, ra, da)
                z_radar = frame_queries.mean(dim=1)
                z_text = text_encoder(texts)

                loss, _ = criterion(z_radar, z_text)
                val_loss_total += loss.item()
                n_val += 1

        val_loss = val_loss_total / max(n_val, 1)
        history["val_loss"].append(val_loss)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{config.stage1.num_epochs} | "
                f"Train Loss: {epoch_losses['total']:.4f} "
                f"(MSE: {epoch_losses['mse']:.4f}, "
                f"Contr: {epoch_losses['contrastive']:.4f}) | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "radar_qformer": radar_qformer.state_dict(),
                "text_encoder": text_encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
            }, os.path.join(output_dir, "best_stage1.pt"))

    # Save final model
    torch.save({
        "epoch": config.stage1.num_epochs,
        "radar_qformer": radar_qformer.state_dict(),
        "text_encoder": text_encoder.state_dict(),
        "val_loss": val_loss,
    }, os.path.join(output_dir, "final_stage1.pt"))

    # Save history
    with open(os.path.join(output_dir, "stage1_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nStage 1 complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {output_dir}")

    return radar_qformer, text_encoder, history


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Representation Alignment")
    parser.add_argument("--data_dir", type=str, default="data/toy_dataset")
    parser.add_argument("--output_dir", type=str, default="output/stage1")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--curriculum_phase", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = FullConfig()
    config.stage1.num_epochs = args.num_epochs
    config.stage1.batch_size = args.batch_size
    config.stage1.lr = args.lr
    config.device = args.device
    config.seed = args.seed

    train_stage1(args.data_dir, args.output_dir, config, args.curriculum_phase)


if __name__ == "__main__":
    main()
