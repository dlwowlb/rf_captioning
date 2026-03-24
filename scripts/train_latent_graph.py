#!/usr/bin/env python3
"""
Latent Graph Dynamics — Stage 1 학습 스크립트

문서 1-2의 파이프라인:
  Radar Point Cloud → Observation Encoding → Posterior Inference
  → Latent Graph Dynamics → Observation Reconstruction
  → Sequence Readout → Motion Alignment

Loss:
  L^(1) = L_obs + β·L_KL + λ_m·L_(r-m)

기존 RadarLLM의 VQ-VAE tokenizer를 대체.
g^radar embedding을 직접 생성하여 downstream에 사용.

사용법:
  python scripts/train_latent_graph.py --config configs/latent_graph.yaml
  python scripts/train_latent_graph.py --config configs/latent_graph.yaml --device cpu
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.latent_graph_dynamics import LatentGraphDynamicsModel
from models.motion_encoder import build_motion_encoder


# ============================================================
# Dataset (기존 generate_pt.py + add_humanml3d.py 호환)
# ============================================================

class RadarMotionDataset(Dataset):
    """
    각 .npz:
      - point_cloud:       (T_radar, 128, 6)
      - text:              str
      - motion_humanml3d:  (T_motion, 263)
      - motion_latent:     (T_motion, 201)
      - motion_joints:     (T_motion, 22, 3)
    """

    def __init__(self, data_dir, max_radar_frames=100,
                 max_motion_frames=300, points_per_frame=128,
                 point_dims=4, motion_input_type="humanml3d"):
        super().__init__()
        self.max_radar_frames = max_radar_frames
        self.max_motion_frames = max_motion_frames
        self.points_per_frame = points_per_frame
        self.point_dims = point_dims
        self.motion_input_type = motion_input_type

        self.samples = sorted(Path(data_dir).glob("*.npz"))
        if not self.samples:
            # 상위 디렉토리의 모든 하위 폴더 검색
            self.samples = sorted(Path(data_dir).rglob("sample_*.npz"))
        print(f"[Dataset] {len(self.samples)} samples, motion={motion_input_type}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx], allow_pickle=True)

        # ── Point cloud ──
        pc = data["point_cloud"].astype(np.float32)
        T_r = min(pc.shape[0], self.max_radar_frames)
        pc = pc[:T_r]

        D = min(pc.shape[-1], self.point_dims)
        pc_padded = np.zeros((self.max_radar_frames, self.points_per_frame,
                              self.point_dims), dtype=np.float32)
        pc_padded[:T_r, :, :D] = pc[:, :, :D]

        radar_mask = np.zeros(self.max_radar_frames, dtype=np.bool_)
        radar_mask[:T_r] = True

        # ── Motion data ──
        motion = self._load_motion(data)
        T_m = min(motion.shape[0], self.max_motion_frames)
        motion = motion[:T_m]

        motion_dim = motion.shape[-1]
        motion_padded = np.zeros((self.max_motion_frames, motion_dim),
                                  dtype=np.float32)
        motion_padded[:T_m] = motion
        motion_mask = np.zeros(self.max_motion_frames, dtype=np.bool_)
        motion_mask[:T_m] = True

        text = str(data["text"])

        # ── Motion joints (for interaction phase pseudo-labels) ──
        joints = None
        if "motion_joints" in data:
            j = data["motion_joints"].astype(np.float32)
            if j.ndim == 3 and j.shape[1] >= 22:
                joints = torch.from_numpy(j[:, :22, :])

        return {
            "point_cloud": torch.from_numpy(pc_padded),
            "radar_mask": torch.from_numpy(radar_mask),
            "motion": torch.from_numpy(motion_padded),
            "motion_mask": torch.from_numpy(motion_mask),
            "motion_joints": joints,  # (T_m, 22, 3) or None
            "text": text,
        }

    def _load_motion(self, data):
        key_map = {
            "humanml3d": "motion_humanml3d",
            "latent": "motion_latent",
            "joints": "motion_joints",
            "rot6d": "motion_rot6d",
        }
        # 1. Config 지정 타입
        preferred = key_map.get(self.motion_input_type)
        if preferred and preferred in data:
            m = data[preferred].astype(np.float32)
            if m.ndim == 3:
                m = m.reshape(m.shape[0], -1)
            return m

        # 2. Fallback
        for key in ["motion_humanml3d", "motion_latent", "motion_joints"]:
            if key in data:
                m = data[key].astype(np.float32)
                if m.ndim == 3:
                    m = m.reshape(m.shape[0], -1)
                return m

        # 3. Empty
        dim_map = {"humanml3d": 263, "latent": 201, "joints": 66, "rot6d": 132}
        dim = dim_map.get(self.motion_input_type, 263)
        return np.zeros((1, dim), dtype=np.float32)


def collate_fn(batch):
    return {
        "point_cloud": torch.stack([b["point_cloud"] for b in batch]),
        "radar_mask": torch.stack([b["radar_mask"] for b in batch]),
        "motion": torch.stack([b["motion"] for b in batch]),
        "motion_mask": torch.stack([b["motion_mask"] for b in batch]),
        "motion_joints": [b["motion_joints"] for b in batch],  # list, may contain None
        "texts": [b["text"] for b in batch],
    }


def _generate_batch_phase_labels(motion_joints_list, T_radar, labeler, device):
    """
    Per-sample pseudo-labels → batched tensors.

    Args:
        motion_joints_list: list of (T_m, 22, 3) tensors or None
        T_radar: int — radar sequence length (pad/truncate target)
        labeler: InteractionPseudoLabeler instance
        device: torch device
    Returns:
        labels:     (B, T_radar) LongTensor or None
        confidence: (B, T_radar) FloatTensor or None
    """
    B = len(motion_joints_list)

    # If no sample has joints, skip phase loss entirely
    if all(j is None for j in motion_joints_list):
        return None, None

    all_labels = []
    all_conf = []

    for joints in motion_joints_list:
        if joints is not None:
            lab, conf = labeler.generate(joints=joints)
            T_lab = lab.shape[0]
            # Pad or truncate to T_radar
            if T_lab >= T_radar:
                lab = lab[:T_radar]
                conf = conf[:T_radar]
            else:
                lab = torch.cat([lab, torch.zeros(T_radar - T_lab, dtype=torch.long)])
                conf = torch.cat([conf, torch.zeros(T_radar - T_lab)])
        else:
            # No joints → default to no_interaction with zero confidence
            lab = torch.zeros(T_radar, dtype=torch.long)
            conf = torch.zeros(T_radar)

        all_labels.append(lab)
        all_conf.append(conf)

    return (torch.stack(all_labels).to(device),
            torch.stack(all_conf).to(device))


# ============================================================
# Diagnostics
# ============================================================

@torch.no_grad()
def log_diagnostics(model, batch, device, epoch):
    """Confidence, node diversity 체크."""
    model.eval()
    pc = batch["point_cloud"].to(device)
    mask = batch["radar_mask"].to(device)

    out = model.forward_sequence(pc, mask)
    node_seq = out["node_history"]              # (B, T, M, D)
    conf_seq = out["confidence"]                # (B, T, M, 1)

    avg_conf = conf_seq.mean().item()
    node_std = node_seq.std(dim=2).mean().item()

    print(f"  [Diag] epoch {epoch}: "
          f"confidence={avg_conf:.3f}, "
          f"node_div={node_std:.4f}")
    model.train()


# ============================================================
# Training Loop
# ============================================================

def train(config, args):
    print("\n" + "=" * 60)
    print("Stage 1: Latent Graph Dynamics Training")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lg_cfg = config["latent_graph"]
    ds_cfg = config.get("dataset", {})
    me_cfg = config.get("motion_encoder", {})
    log_cfg = config.get("logging", {})

    # ── Model ──
    model = LatentGraphDynamicsModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Latent Graph Model: {total_params:,} params")
    print(f"  Nodes: {model.num_nodes}, dim: {model.node_dim}")
    print(f"  Context: {model.ctx_dim}D, history K={model.ctx_history_len}")
    print(f"  Output: {model.out_dim}D")

    # ── Motion Encoder (for L_(r-m)) ──
    motion_enc = build_motion_encoder(config).to(device)

    # ── Optimizer ──
    all_params = list(model.parameters())
    all_params += [p for p in motion_enc.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        all_params,
        lr=float(lg_cfg["lr"]),
        weight_decay=float(lg_cfg.get("weight_decay", 1e-2)),
    )

    # ── Scheduler ──
    sched_type = lg_cfg.get("scheduler", "cosine")
    warmup = lg_cfg.get("warmup_epochs", 5)
    total_epochs = lg_cfg["epochs"]

    if sched_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup)
    else:
        scheduler = None

    # Warmup
    warmup_scheduler = None
    if warmup > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup)

    # ── Data ──
    mi_type = me_cfg.get("input_type", "humanml3d")
    point_dims = lg_cfg.get("point_in_dim", 4)

    train_ds = RadarMotionDataset(
        ds_cfg["train_dir"],
        max_radar_frames=ds_cfg.get("max_seq_len", 100),
        max_motion_frames=ds_cfg.get("max_motion_frames", 300),
        points_per_frame=ds_cfg.get("points_per_frame", 128),
        point_dims=point_dims,
        motion_input_type=mi_type,
    )
    val_ds = RadarMotionDataset(
        ds_cfg["val_dir"],
        max_radar_frames=ds_cfg.get("max_seq_len", 100),
        max_motion_frames=ds_cfg.get("max_motion_frames", 300),
        points_per_frame=ds_cfg.get("points_per_frame", 128),
        point_dims=point_dims,
        motion_input_type=mi_type,
    )
    train_loader = DataLoader(
        train_ds, batch_size=lg_cfg["batch_size"],
        shuffle=True, collate_fn=collate_fn, num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=lg_cfg["batch_size"],
        shuffle=False, collate_fn=collate_fn, num_workers=2,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_val = float("inf")
    history = []

    # ── Training ──
    from models.latent_graph_dynamics import InteractionPseudoLabeler
    phase_labeler = InteractionPseudoLabeler()
    



    for epoch in range(total_epochs):
        model.train()
        motion_enc.train()

        loss_keys = ["total", "obs", "kl", "rm", "phase"]
        losses = {k: 0.0 for k in loss_keys}
        num_batches = 0

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1}/{total_epochs}")
        for batch in pbar:
            pc = batch["point_cloud"].to(device)
            radar_mask = batch["radar_mask"].to(device)
            motion = batch["motion"].to(device)

            F_mot = motion_enc(motion)  # (B, T_mot, feat_dim)
            #g_motion = F_mot.mean(dim=1)  # (B, feat_dim)
            m_mask = batch["motion_mask"].to(device).float()  # (B, T_orig)

            # ★ adaptive pooling으로 mask를 F_mot 길이에 맞춤
            T_out = F_mot.shape[1]
            m_mask = F.adaptive_max_pool1d(m_mask.unsqueeze(1), T_out).squeeze(1)  # (B, T')

            g_motion = (F_mot * m_mask.unsqueeze(-1)).sum(dim=1) / m_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

            # ── Generate interaction phase pseudo-labels ──
            T_radar = pc.shape[1]
            phase_labels, phase_conf = _generate_batch_phase_labels(
                batch["motion_joints"], T_radar, phase_labeler, device)

            out = model(pc, motion_features=g_motion,
                        temporal_mask=radar_mask,
                        phase_labels=phase_labels,
                        phase_confidence=phase_conf)
            
            
            optimizer.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                all_params, lg_cfg.get("grad_clip", 1.0))
            optimizer.step()

            losses["total"] += out["loss"].item()
            for k in loss_keys[1:]:
                losses[k] += out[f"loss_{k}"].item()
            num_batches += 1

            if num_batches % log_cfg.get("log_every", 10) == 0:
                pbar.set_postfix(
                    loss=f"{out['loss'].item():.4f}",
                    obs=f"{out['loss_obs'].item():.4f}",
                    phase=f"{out['loss_phase'].item():.4f}",
                )

        # ── Scheduler step ──
        if epoch < warmup and warmup_scheduler is not None:
            warmup_scheduler.step()
        elif scheduler is not None:
            scheduler.step()

        # ── Validation ──
        model.eval()
        motion_enc.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                pc = batch["point_cloud"].to(device)
                mask = batch["radar_mask"].to(device)
                motion = batch["motion"].to(device)

                F_mot = motion_enc(motion)  # (B, T_mot, feat_dim)
                #g_motion = F_mot.mean(dim=1)  # (B, feat_dim)
                m_mask = batch["motion_mask"].to(device).float()  # (B, T_orig)

                # ★ adaptive pooling으로 mask를 F_mot 길이에 맞춤
                T_out = F_mot.shape[1]
                m_mask = F.adaptive_max_pool1d(m_mask.unsqueeze(1), T_out).squeeze(1)  # (B, T')

                g_motion = (F_mot * m_mask.unsqueeze(-1)).sum(dim=1) / m_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

                T_radar = pc.shape[1]
                phase_labels, phase_conf = _generate_batch_phase_labels(
                    batch["motion_joints"], T_radar, phase_labeler, device)

                out = model(pc, motion_features=g_motion,
                            temporal_mask=mask,
                            phase_labels=phase_labels,
                            phase_confidence=phase_conf)
                val_loss += out["loss"].item()
                val_batches += 1

        val_avg = val_loss / max(val_batches, 1)
        n = max(num_batches, 1)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}: "
              f"loss={losses['total']/n:.4f} "
              f"(obs={losses['obs']/n:.4f} "
              f"kl={losses['kl']/n:.4f} "
              f"rm={losses['rm']/n:.4f} "
              f"phase={losses['phase']/n:.4f}) "
              f"val={val_avg:.4f} lr={lr:.2e}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": losses["total"] / n,
            "train_obs": losses["obs"] / n,
            "train_kl": losses["kl"] / n,
            "train_rm": losses["rm"] / n,
            "train_phase": losses["phase"] / n,
            "val_loss": val_avg,
            "lr": lr,
        })

        # ── Diagnostics ──
        if (epoch + 1) % log_cfg.get("vis_every", 20) == 0:
            sample_batch = next(iter(val_loader))
            log_diagnostics(model, sample_batch, device, epoch + 1)

        # ── Save ──
        if val_avg < best_val:
            best_val = val_avg
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "motion_enc_state_dict": motion_enc.state_dict(),
                "val_loss": val_avg,
                "config": config,
            }, os.path.join(args.output_dir, "latent_graph_best.pt"))
            print(f"  ★ Best model saved (val={val_avg:.4f})")

        if (epoch + 1) % log_cfg.get("save_every", 10) == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "motion_enc_state_dict": motion_enc.state_dict(),
            }, os.path.join(args.output_dir, f"latent_graph_ep{epoch+1}.pt"))

    # ── Final save ──
    torch.save({
        "model_state_dict": model.state_dict(),
        "motion_enc_state_dict": motion_enc.state_dict(),
        "config": config,
    }, os.path.join(args.output_dir, "latent_graph_final.pt"))

    # ── History ──
    with open(os.path.join(args.output_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best val loss: {best_val:.4f}")
    print(f"  Checkpoints: {args.output_dir}")
    print(f"{'='*60}")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Latent Graph Dynamics Stage 1 Training")
    parser.add_argument("--config", default="configs/latent_graph.yaml")
    parser.add_argument("--output_dir", default="checkpoints/latent_graph")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    train(config, args)