#!/usr/bin/env python3
"""
RadarCaptioner Training — v8.1 (Qwen3-8B QLoRA)

3단계 학습:
  Stage 1: LatentGraph pretrain (201D MSE만)
  Stage 2: Captioner 학습 (LatentGraph frozen, projection + Qwen LoRA)
  Stage 3: Joint fine-tune (전체, 낮은 LR)

사용법:
  python pipelines/train_captioner.py --stage all --config configs/captioner.yaml
  python pipelines/train_captioner.py --stage pretrain_lg --config configs/captioner.yaml
  python pipelines/train_captioner.py --stage caption --lg_ckpt checkpoints/captioner/stage1/best.pt

  python pipelines/train_captioner.py --stage caption --lg_ckpt checkpoints/latent_graph2/latent_graph_best.pt

  python pipelines/train_captioner.py --stage joint --caption_ckpt checkpoints/captioner/stage2/final.pt
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models2.radar_captioner import RadarCaptioner
from models2.latent_graph_dynamics import (
    LatentGraphDynamicsModel,
    InteractionPseudoLabeler,
    NodeDiagnostics,
)


# ============================================================
# Dataset
# ============================================================

class RadarCaptionDataset(Dataset):
    def __init__(self, data_dir, max_radar_frames=100,
                 max_motion_frames=300, points_per_frame=128,
                 point_dims=4):
        super().__init__()
        self.max_radar_frames = max_radar_frames
        self.max_motion_frames = max_motion_frames
        self.points_per_frame = points_per_frame
        self.point_dims = point_dims

        self.samples = sorted(Path(data_dir).glob("*.npz"))
        if not self.samples:
            self.samples = sorted(Path(data_dir).rglob("sample_*.npz"))
        print(f"[Dataset] {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx], allow_pickle=True)

        # Radar
        pc = data["point_cloud"].astype(np.float32)
        T_r = min(pc.shape[0], self.max_radar_frames)
        D = min(pc.shape[-1], self.point_dims)
        pc_padded = np.zeros((self.max_radar_frames, self.points_per_frame,
                              self.point_dims), dtype=np.float32)
        pc_padded[:T_r, :, :D] = pc[:T_r, :, :D]
        radar_mask = np.zeros(self.max_radar_frames, dtype=np.bool_)
        radar_mask[:T_r] = True

        # 201D latent
        if "motion_latent" in data:
            latent = data["motion_latent"].astype(np.float32)
        else:
            latent = np.zeros((1, 201), dtype=np.float32)
        T_m = min(latent.shape[0], self.max_motion_frames)
        latent_padded = np.zeros((self.max_motion_frames, 201), dtype=np.float32)
        latent_padded[:T_m] = latent[:T_m]

        # Joints (phase label용)
        joints = None
        if "motion_joints" in data:
            j = data["motion_joints"].astype(np.float32)
            if j.ndim == 3 and j.shape[1] >= 22:
                joints = torch.from_numpy(j[:, :22, :])

        text = str(data["text"])

        return {
            "point_cloud": torch.from_numpy(pc_padded),
            "radar_mask": torch.from_numpy(radar_mask),
            "motion_latent_201": torch.from_numpy(latent_padded),
            "motion_joints": joints,
            "text": text,
        }


def collate_fn(batch):
    return {
        "point_cloud": torch.stack([b["point_cloud"] for b in batch]),
        "radar_mask": torch.stack([b["radar_mask"] for b in batch]),
        "motion_latent_201": torch.stack([b["motion_latent_201"] for b in batch]),
        "motion_joints": [b["motion_joints"] for b in batch],
        "texts": [b["text"] for b in batch],
    }


# ============================================================
# Helpers
# ============================================================

def _generate_batch_phase_labels(motion_joints_list, T_radar, labeler, device):
    B = len(motion_joints_list)
    if all(j is None for j in motion_joints_list):
        return None, None
    all_labels, all_conf = [], []
    for joints in motion_joints_list:
        if joints is not None:
            lab, conf = labeler.generate(joints=joints)
            T_lab = lab.shape[0]
            if T_lab >= T_radar:
                lab, conf = lab[:T_radar], conf[:T_radar]
            else:
                lab = torch.cat([lab, torch.zeros(T_radar - T_lab, dtype=torch.long)])
                conf = torch.cat([conf, torch.zeros(T_radar - T_lab)])
        else:
            lab = torch.zeros(T_radar, dtype=torch.long)
            conf = torch.zeros(T_radar)
        all_labels.append(lab)
        all_conf.append(conf)
    return torch.stack(all_labels).to(device), torch.stack(all_conf).to(device)


@torch.no_grad()
def run_test_captioning(model, loader, device, epoch, max_samples=5):
    model.eval()
    print(f"\n{'─'*60}")
    print(f"  Epoch {epoch} — Test Captioning (Qwen3-8B)")
    print(f"{'─'*60}")

    count = 0
    for batch in loader:
        if count >= max_samples:
            break
        pc = batch["point_cloud"].to(device)
        mask = batch["radar_mask"].to(device)

        captions = model.generate(pc, mask, max_new_tokens=64)

        for j in range(len(batch["texts"])):
            if count >= max_samples:
                break
            print(f"  [{count+1}] GT:   \"{batch['texts'][j]}\"")
            print(f"       Pred: \"{captions[j]}\"")
            print()
            count += 1

    print(f"{'─'*60}\n")
    model.train()


# ============================================================
# Stage 1: LatentGraph Pretrain (201D MSE)
# ============================================================

def train_stage1(config, args):
    print("\n" + "=" * 60)
    print("Stage 1: LatentGraph Pretrain (201D MSE)")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lg_cfg = config["latent_graph"]
    ds_cfg = config.get("dataset", {})
    stage_cfg = config.get("stage1", {})

    model = LatentGraphDynamicsModel(config).to(device)
    print(f"LatentGraph: {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(stage_cfg.get("lr", lg_cfg.get("lr", 3e-4))),
        weight_decay=float(lg_cfg.get("weight_decay", 1e-2)))

    epochs = stage_cfg.get("epochs", 100)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = RadarCaptionDataset(
        ds_cfg["train_dir"],
        max_radar_frames=ds_cfg.get("max_seq_len", 100),
        max_motion_frames=ds_cfg.get("max_motion_frames", 300),
        points_per_frame=ds_cfg.get("points_per_frame", 128),
        point_dims=lg_cfg.get("point_in_dim", 4))
    val_ds = RadarCaptionDataset(
        ds_cfg["val_dir"],
        max_radar_frames=ds_cfg.get("max_seq_len", 100),
        max_motion_frames=ds_cfg.get("max_motion_frames", 300),
        points_per_frame=ds_cfg.get("points_per_frame", 128),
        point_dims=lg_cfg.get("point_in_dim", 4))

    train_loader = DataLoader(train_ds, batch_size=stage_cfg.get("batch_size", 4),
                              shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=stage_cfg.get("batch_size", 4),
                            shuffle=False, collate_fn=collate_fn, num_workers=2)

    phase_labeler = InteractionPseudoLabeler()
    best_val = float("inf")
    save_dir = os.path.join(args.output_dir, "stage1")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        losses = {"total": 0, "obs": 0, "kl": 0, "latent": 0}
        n = 0

        for batch in tqdm(train_loader, desc=f"S1 {epoch+1}/{epochs}"):
            pc = batch["point_cloud"].to(device)
            mask = batch["radar_mask"].to(device)
            latent = batch["motion_latent_201"].to(device)

            T_radar = pc.shape[1]
            phase_labels, phase_conf = _generate_batch_phase_labels(
                batch["motion_joints"], T_radar, phase_labeler, device)

            out = model(pc, motion_latent_201=latent, temporal_mask=mask,
                        phase_labels=phase_labels, phase_confidence=phase_conf)

            if torch.isnan(out["loss"]):
                continue

            optimizer.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in losses:
                key = "loss" if k == "total" else f"loss_{k}"
                losses[k] += out[key].item()
            n += 1

        scheduler.step()

        model.eval()
        val_loss = 0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                pc = batch["point_cloud"].to(device)
                mask = batch["radar_mask"].to(device)
                latent = batch["motion_latent_201"].to(device)
                out = model(pc, motion_latent_201=latent, temporal_mask=mask)
                if not torch.isnan(out["loss"]):
                    val_loss += out["loss"].item()
                    vn += 1
        val_avg = val_loss / max(vn, 1)
        m = max(n, 1)
        print(f"Ep {epoch+1}: loss={losses['total']/m:.4f} "
              f"(obs={losses['obs']/m:.4f} lat={losses['latent']/m:.6f}) "
              f"val={val_avg:.4f}")

        if val_avg < best_val:
            best_val = val_avg
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "val_loss": val_avg},
                       os.path.join(save_dir, "best.pt"))

    torch.save({"model_state_dict": model.state_dict()},
               os.path.join(save_dir, "final.pt"))
    print(f"Stage 1 완료: {save_dir}")
    return os.path.join(save_dir, "best.pt")


# ============================================================
# Stage 2: Captioner (LatentGraph frozen, Qwen LoRA + projection)
# ============================================================

def train_stage2(config, args, lg_ckpt=None):
    print("\n" + "=" * 60)
    print("Stage 2: Captioner (Qwen3-8B QLoRA, LatentGraph frozen)")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds_cfg = config.get("dataset", {})
    lg_cfg = config.get("latent_graph", {})
    stage_cfg = config.get("stage2", {})

    # Full model (Qwen 4-bit 로드 포함)
    model = RadarCaptioner(config)
    # Note: Qwen은 device_map="auto"로 자동 배치됨
    # LatentGraph만 명시적으로 device로 이동
    model.latent_graph = model.latent_graph.to(device)
    model.token_proj = model.token_proj.to(device)
    model.global_proj = model.global_proj.to(device)
    model.task_prefix = nn.Parameter(model.task_prefix.to(device))

    # Load pretrained LatentGraph
    lg_ckpt = lg_ckpt or args.lg_ckpt
    if lg_ckpt and os.path.exists(lg_ckpt):
        ckpt = torch.load(lg_ckpt, map_location=device, weights_only=False)
        model.latent_graph.load_state_dict(
            ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"LatentGraph loaded: {lg_ckpt}")

    # Freeze LatentGraph
    model.freeze_latent_graph()
    model.print_param_stats()

    # Optimizer (Qwen LoRA + projection만)
    trainable = model.get_trainable_params()
    optimizer = optim.AdamW(
        trainable,
        lr=float(stage_cfg.get("lr", 1e-4)),
        weight_decay=float(stage_cfg.get("weight_decay", 1e-2)))

    epochs = stage_cfg.get("epochs", 50)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = RadarCaptionDataset(
        ds_cfg["train_dir"],
        max_radar_frames=ds_cfg.get("max_seq_len", 100),
        max_motion_frames=ds_cfg.get("max_motion_frames", 300),
        points_per_frame=ds_cfg.get("points_per_frame", 128),
        point_dims=lg_cfg.get("point_in_dim", 4))
    val_ds = RadarCaptionDataset(
        ds_cfg["val_dir"],
        max_radar_frames=ds_cfg.get("max_seq_len", 100),
        max_motion_frames=ds_cfg.get("max_motion_frames", 300),
        points_per_frame=ds_cfg.get("points_per_frame", 128),
        point_dims=lg_cfg.get("point_in_dim", 4))

    train_loader = DataLoader(train_ds, batch_size=stage_cfg.get("batch_size", 2),
                              shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=stage_cfg.get("batch_size", 2),
                            shuffle=False, collate_fn=collate_fn, num_workers=2)

    best_val = float("inf")
    save_dir = os.path.join(args.output_dir, "stage2")
    os.makedirs(save_dir, exist_ok=True)
    caption_every = stage_cfg.get("caption_every", 5)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_cap = 0
        n = 0

        for batch in tqdm(train_loader, desc=f"S2 {epoch+1}/{epochs}"):
            pc = batch["point_cloud"].to(device)
            mask = batch["radar_mask"].to(device)
            latent = batch["motion_latent_201"].to(device)
            texts = batch["texts"]

            out = model(pc, mask, texts, motion_latent_201=latent)

            if torch.isnan(out["loss"]):
                continue

            optimizer.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            epoch_loss += out["loss"].item()
            epoch_cap += out["loss_caption"].item()
            n += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        val_cap = 0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                pc = batch["point_cloud"].to(device)
                mask = batch["radar_mask"].to(device)
                latent = batch["motion_latent_201"].to(device)
                out = model(pc, mask, batch["texts"], motion_latent_201=latent)
                if not torch.isnan(out["loss"]):
                    val_loss += out["loss"].item()
                    val_cap += out["loss_caption"].item()
                    vn += 1
        val_avg = val_loss / max(vn, 1)
        val_cap_avg = val_cap / max(vn, 1)
        m = max(n, 1)

        print(f"Ep {epoch+1}: loss={epoch_loss/m:.4f} "
              f"cap={epoch_cap/m:.4f} "
              f"val={val_avg:.4f} val_cap={val_cap_avg:.4f}")

        if (epoch + 1) % caption_every == 0:
            run_test_captioning(model, val_loader, device, epoch + 1)

        if val_avg < best_val:
            best_val = val_avg
            # LoRA adapter만 저장 (Qwen base는 이미 있으므로)
            save_dict = {
                "epoch": epoch,
                "latent_graph_state": model.latent_graph.state_dict(),
                "token_proj_state": model.token_proj.state_dict(),
                "global_proj_state": model.global_proj.state_dict(),
                "task_prefix": model.task_prefix.data,
                "val_loss": val_avg,
                "val_caption": val_cap_avg,
            }
            # LoRA weights
            try:
                model.qwen.save_pretrained(os.path.join(save_dir, "qwen_lora"))
                save_dict["qwen_lora_path"] = os.path.join(save_dir, "qwen_lora")
            except Exception as e:
                print(f"  LoRA save warning: {e}")
                save_dict["qwen_state"] = {
                    k: v for k, v in model.qwen.state_dict().items()
                    if (v.requires_grad if hasattr(v, 'requires_grad') else True)
                }

            torch.save(save_dict, os.path.join(save_dir, "best.pt"))
            print(f"  ★ Best (val={val_avg:.4f}, cap={val_cap_avg:.4f})")

    torch.save(save_dict, os.path.join(save_dir, "final.pt"))
    print(f"Stage 2 완료: {save_dir}")

    run_test_captioning(model, val_loader, device, epochs, max_samples=10)
    return os.path.join(save_dir, "best.pt")


# ============================================================
# Stage 3: Joint Fine-tune
# ============================================================

def train_stage3(config, args, caption_ckpt=None):
    print("\n" + "=" * 60)
    print("Stage 3: Joint Fine-tune (all parameters)")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds_cfg = config.get("dataset", {})
    lg_cfg = config.get("latent_graph", {})
    stage_cfg = config.get("stage3", {})

    model = RadarCaptioner(config)
    model.latent_graph = model.latent_graph.to(device)
    model.token_proj = model.token_proj.to(device)
    model.global_proj = model.global_proj.to(device)
    model.task_prefix = nn.Parameter(model.task_prefix.to(device))

    # Load stage2 checkpoint
    caption_ckpt = caption_ckpt or args.caption_ckpt
    if caption_ckpt and os.path.exists(caption_ckpt):
        ckpt = torch.load(caption_ckpt, map_location=device, weights_only=False)
        if "latent_graph_state" in ckpt:
            model.latent_graph.load_state_dict(ckpt["latent_graph_state"], strict=False)
        if "token_proj_state" in ckpt:
            model.token_proj.load_state_dict(ckpt["token_proj_state"])
        if "global_proj_state" in ckpt:
            model.global_proj.load_state_dict(ckpt["global_proj_state"])
        if "task_prefix" in ckpt:
            model.task_prefix.data.copy_(ckpt["task_prefix"])
        # LoRA weights
        lora_path = ckpt.get("qwen_lora_path")
        if lora_path and os.path.exists(lora_path):
            try:
                from peft import PeftModel
                model.qwen = PeftModel.from_pretrained(model.qwen, lora_path)
                print(f"LoRA loaded: {lora_path}")
            except Exception as e:
                print(f"LoRA load warning: {e}")
        print(f"Stage2 loaded: {caption_ckpt}")

    # Unfreeze LatentGraph
    model.unfreeze_latent_graph()
    model.print_param_stats()

    # Separate LR per component
    optimizer = optim.AdamW([
        {"params": model.latent_graph.parameters(),
         "lr": float(stage_cfg.get("lg_lr", 1e-5))},
        {"params": list(model.token_proj.parameters())
                   + list(model.global_proj.parameters())
                   + [model.task_prefix],
         "lr": float(stage_cfg.get("proj_lr", 5e-5))},
        {"params": [p for p in model.qwen.parameters() if p.requires_grad],
         "lr": float(stage_cfg.get("qwen_lr", 3e-5))},
    ], weight_decay=float(stage_cfg.get("weight_decay", 1e-2)))

    epochs = stage_cfg.get("epochs", 30)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    phase_labeler = InteractionPseudoLabeler()

    train_ds = RadarCaptionDataset(
        ds_cfg["train_dir"],
        max_radar_frames=ds_cfg.get("max_seq_len", 100),
        max_motion_frames=ds_cfg.get("max_motion_frames", 300),
        points_per_frame=ds_cfg.get("points_per_frame", 128),
        point_dims=lg_cfg.get("point_in_dim", 4))
    val_ds = RadarCaptionDataset(
        ds_cfg["val_dir"],
        max_radar_frames=ds_cfg.get("max_seq_len", 100),
        max_motion_frames=ds_cfg.get("max_motion_frames", 300),
        points_per_frame=ds_cfg.get("points_per_frame", 128),
        point_dims=lg_cfg.get("point_in_dim", 4))

    train_loader = DataLoader(train_ds, batch_size=stage_cfg.get("batch_size", 2),
                              shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=stage_cfg.get("batch_size", 2),
                            shuffle=False, collate_fn=collate_fn, num_workers=2)

    best_val = float("inf")
    save_dir = os.path.join(args.output_dir, "stage3")
    os.makedirs(save_dir, exist_ok=True)
    caption_every = stage_cfg.get("caption_every", 3)

    for epoch in range(epochs):
        model.train()
        losses = {"total": 0, "obs": 0, "latent": 0, "caption": 0}
        n = 0

        for batch in tqdm(train_loader, desc=f"S3 {epoch+1}/{epochs}"):
            pc = batch["point_cloud"].to(device)
            mask = batch["radar_mask"].to(device)
            latent = batch["motion_latent_201"].to(device)

            T_radar = pc.shape[1]
            phase_labels, phase_conf = _generate_batch_phase_labels(
                batch["motion_joints"], T_radar, phase_labeler, device)

            out = model(pc, mask, batch["texts"],
                        motion_latent_201=latent,
                        phase_labels=phase_labels,
                        phase_confidence=phase_conf)

            if torch.isnan(out["loss"]):
                continue

            optimizer.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in losses:
                key = "loss" if k == "total" else f"loss_{k}"
                losses[k] += out[key].item()
            n += 1

        scheduler.step()

        model.eval()
        val_loss = 0
        val_cap = 0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                pc = batch["point_cloud"].to(device)
                mask = batch["radar_mask"].to(device)
                latent = batch["motion_latent_201"].to(device)
                out = model(pc, mask, batch["texts"], motion_latent_201=latent)
                if not torch.isnan(out["loss"]):
                    val_loss += out["loss"].item()
                    val_cap += out["loss_caption"].item()
                    vn += 1
        val_avg = val_loss / max(vn, 1)
        val_cap_avg = val_cap / max(vn, 1)
        m = max(n, 1)

        print(f"Ep {epoch+1}: loss={losses['total']/m:.4f} "
              f"(obs={losses['obs']/m:.4f} lat={losses['latent']/m:.6f} "
              f"cap={losses['caption']/m:.4f}) "
              f"val={val_avg:.4f} cap={val_cap_avg:.4f}")

        if (epoch + 1) % caption_every == 0:
            run_test_captioning(model, val_loader, device, epoch + 1)

        if val_avg < best_val:
            best_val = val_avg
            torch.save({
                "epoch": epoch,
                "latent_graph_state": model.latent_graph.state_dict(),
                "token_proj_state": model.token_proj.state_dict(),
                "global_proj_state": model.global_proj.state_dict(),
                "task_prefix": model.task_prefix.data,
                "val_loss": val_avg,
            }, os.path.join(save_dir, "best.pt"))
            try:
                model.qwen.save_pretrained(os.path.join(save_dir, "qwen_lora"))
            except:
                pass
            print(f"  ★ Best (val={val_avg:.4f}, cap={val_cap_avg:.4f})")

    # Final predictions
    _save_all_predictions(model, val_loader, device, save_dir)
    print(f"Stage 3 완료: {save_dir}")


@torch.no_grad()
def _save_all_predictions(model, loader, device, save_dir):
    model.eval()
    results = []
    for batch in tqdm(loader, desc="Final predictions"):
        pc = batch["point_cloud"].to(device)
        mask = batch["radar_mask"].to(device)
        captions = model.generate(pc, mask, max_new_tokens=64)
        for gt, pred in zip(batch["texts"], captions):
            results.append({"ground_truth": gt, "prediction": pred})

    path = os.path.join(save_dir, "predictions.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Predictions: {path} ({len(results)} samples)")
    for r in results[:5]:
        print(f"  GT:   \"{r['ground_truth']}\"")
        print(f"  Pred: \"{r['prediction']}\"")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="RadarCaptioner Training v8.1 (Qwen3-8B QLoRA)")
    parser.add_argument("--stage", required=True,
                        choices=["pretrain_lg", "caption", "joint", "all"])
    parser.add_argument("--config", default="configs/captioner.yaml")
    parser.add_argument("--output_dir", default="checkpoints/captioner")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lg_ckpt", default=None)
    parser.add_argument("--caption_ckpt", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.stage == "pretrain_lg":
        train_stage1(config, args)
    elif args.stage == "caption":
        train_stage2(config, args)
    elif args.stage == "joint":
        train_stage3(config, args)
    elif args.stage == "all":
        lg_ckpt = train_stage1(config, args)
        cap_ckpt = train_stage2(config, args, lg_ckpt=lg_ckpt)
        train_stage3(config, args, caption_ckpt=cap_ckpt)

    print("\n학습 완료!")


if __name__ == "__main__":
    main()
