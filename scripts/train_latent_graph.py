#!/usr/bin/env python3
"""
Latent Graph Dynamics — v7.0 Training Script

v6.0 → v7.0 changes:
  [new] motion_seq + motion_mask를 forward에 직접 전달 (mean pooling 제거)
  [new] loss_rm_global + loss_rm_token 이중 contrastive 로깅
  [new] MotionSemanticHead가 motion sequence에서 K개 token 추출
  [new] NodeAwareReadout: node별 temporal pool → K semantic tokens

Loss: L_obs + β·L_KL + λ_m·L_rm_global + λ_t·L_rm_token
      [+ λ_ph·L_phase] [+ λ_d·L_div]

Usage:
  python scripts/train_latent_graph.py --config configs/latent_graph.yaml
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.latent_graph_dynamics import (
    LatentGraphDynamicsModel,
    InteractionPseudoLabeler,
    NodeDiagnostics,
)
from models.motion_encoder import build_motion_encoder


# ============================================================
# Dataset
# ============================================================

class RadarMotionDataset(Dataset):
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
            self.samples = sorted(Path(data_dir).rglob("sample_*.npz"))
        print(f"[Dataset] {len(self.samples)} samples, motion={motion_input_type}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx], allow_pickle=True)

        pc = data["point_cloud"].astype(np.float32)
        T_r = min(pc.shape[0], self.max_radar_frames)
        pc = pc[:T_r]
        D = min(pc.shape[-1], self.point_dims)
        pc_padded = np.zeros((self.max_radar_frames, self.points_per_frame,
                              self.point_dims), dtype=np.float32)
        pc_padded[:T_r, :, :D] = pc[:, :, :D]
        radar_mask = np.zeros(self.max_radar_frames, dtype=np.bool_)
        radar_mask[:T_r] = True

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
            "motion_joints": joints,
            "text": text,
        }

    def _load_motion(self, data):
        key_map = {
            "humanml3d": "motion_humanml3d", "latent": "motion_latent",
            "joints": "motion_joints", "rot6d": "motion_rot6d",
        }
        preferred = key_map.get(self.motion_input_type)
        if preferred and preferred in data:
            m = data[preferred].astype(np.float32)
            return m.reshape(m.shape[0], -1) if m.ndim == 3 else m
        for key in ["motion_humanml3d", "motion_latent", "motion_joints"]:
            if key in data:
                m = data[key].astype(np.float32)
                return m.reshape(m.shape[0], -1) if m.ndim == 3 else m
        dim_map = {"humanml3d": 263, "latent": 201, "joints": 66, "rot6d": 132}
        dim = dim_map.get(self.motion_input_type, 263)
        return np.zeros((1, dim), dtype=np.float32)


def collate_fn(batch):
    return {
        "point_cloud": torch.stack([b["point_cloud"] for b in batch]),
        "radar_mask": torch.stack([b["radar_mask"] for b in batch]),
        "motion": torch.stack([b["motion"] for b in batch]),
        "motion_mask": torch.stack([b["motion_mask"] for b in batch]),
        "motion_joints": [b["motion_joints"] for b in batch],
        "texts": [b["text"] for b in batch],
    }


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


def _align_motion_mask(motion_mask_bool, T_mot, device):
    """motion_enc 출력 길이와 mask 길이 맞추기."""
    T_mask = motion_mask_bool.shape[1]
    if T_mask > T_mot:
        return motion_mask_bool[:, :T_mot]
    elif T_mask < T_mot:
        B = motion_mask_bool.shape[0]
        pad = torch.zeros(B, T_mot - T_mask, dtype=torch.bool, device=device)
        return torch.cat([motion_mask_bool, pad], dim=1)
    return motion_mask_bool


# ============================================================
# Diagnostics
# ============================================================

@torch.no_grad()
def run_full_diagnostics(model, batch, device, epoch, step=0):
    model.eval()
    pc = batch["point_cloud"].to(device)
    mask = batch["radar_mask"].to(device)

    out = model.forward_sequence(pc, mask)
    node_report = model.get_diagnostics(out, mask)
    NodeDiagnostics.log(node_report, epoch, step)

    node_seq = out["node_history"]
    conf_seq = out["confidence"]
    B, T, M, D = node_seq.shape

    # Query diversity
    queries = model.node_queries.squeeze(0)
    q_norm = F.normalize(queries, dim=-1)
    q_sim = (q_norm @ q_norm.t())
    q_eye = torch.eye(M, device=q_sim.device)
    q_off = (q_sim * (1 - q_eye)).abs().mean()
    node_report["query_cosine_sim"] = float(q_off.item())

    # Per-node mean confidence
    conf_per_node = conf_seq.squeeze(-1).mean(dim=(0, 1))
    node_report["per_node_confidence"] = conf_per_node.tolist()

    # Temporal role consistency
    argmax_nodes = conf_seq.squeeze(-1).argmax(dim=2)
    if T > 1:
        same_node = (argmax_nodes[:, 1:] == argmax_nodes[:, :-1]).float().mean()
    else:
        same_node = torch.tensor(1.0)
    node_report["temporal_role_consistency"] = float(same_node.item())

    # Context temporal std
    ctx = out["context_history"]
    ctx_std = ctx.std(dim=1).mean()
    node_report["context_temporal_std"] = float(ctx_std.item())

    # Token attention diversity (v7.0 신규)
    if "token_attn" in out and out["token_attn"] is not None:
        attn = out["token_attn"]  # (B, K, M)
        # 각 token이 다른 node에 attend하는지
        attn_entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1).mean()
        node_report["token_attn_entropy"] = float(attn_entropy.item())
        # token 간 attention 패턴 유사도
        attn_norm = F.normalize(attn, dim=-1)  # (B, K, M)
        tok_sim = torch.bmm(attn_norm, attn_norm.transpose(1, 2))  # (B, K, K)
        K = attn.shape[1]
        tok_eye = torch.eye(K, device=tok_sim.device).unsqueeze(0)
        tok_off = (tok_sim * (1 - tok_eye)).abs().mean()
        node_report["token_pattern_similarity"] = float(tok_off.item())

    print(f"  [NodeDiag Extended] ep{epoch}:")
    print(f"    query_cos_sim={q_off:.4f}")
    print(f"    per_node_conf={[f'{c:.3f}' for c in conf_per_node.tolist()]}")
    print(f"    temporal_role_consistency={node_report['temporal_role_consistency']:.3f}")
    print(f"    context_temporal_std={ctx_std:.4f}")
    if "token_attn_entropy" in node_report:
        print(f"    token_attn_entropy={node_report['token_attn_entropy']:.4f} "
              f"token_pattern_sim={node_report.get('token_pattern_similarity', 0):.4f}")

    if node_report["node_cosine_sim"] > 0.85 and epoch >= 5:
        print(f"  ⚠⚠⚠ WARNING: Node collapse at epoch {epoch}! "
              f"cos_sim={node_report['node_cosine_sim']:.4f}")

    model.train()
    return node_report


# ============================================================
# Training Loop
# ============================================================

def train(config, args):
    print("\n" + "=" * 60)
    print("Stage 1: Latent Graph Dynamics Training — v7.0")
    print("  NodeAwareReadout + Token-Level Contrastive")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lg_cfg = config["latent_graph"]
    ds_cfg = config.get("dataset", {})
    me_cfg = config.get("motion_encoder", {})
    log_cfg = config.get("logging", {})

    # ── Model ──
    model = LatentGraphDynamicsModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params")
    print(f"  Nodes: {model.num_nodes}, dim: {model.node_dim}")
    print(f"  Semantic queries: {model.num_semantic_queries}")
    print(f"  Output: {model.out_dim}D")
    print(f"  Loss: L_obs + {model.beta_kl}·L_KL "
          f"+ {model.lambda_motion}·L_rm_global "
          f"+ {model.lambda_token}·L_rm_token"
          f"{f' + {model.lambda_diversity}·L_div' if model.lambda_diversity > 0 else ''}")
    print(f"  Contrastive temp: {model.contrastive_temp}")

    # Initial query check
    queries = model.node_queries.squeeze(0)
    q_norm = F.normalize(queries, dim=-1)
    q_sim = (q_norm @ q_norm.t())
    q_eye = torch.eye(model.num_nodes, device=q_sim.device)
    init_sim = (q_sim * (1 - q_eye)).abs().mean()
    print(f"  Initial query cos_sim: {init_sim:.4f}")

    # ── Motion Encoder ──
    motion_enc = build_motion_encoder(config).to(device)

    # ── Optimizer ──
    all_params = list(model.parameters())
    all_params += [p for p in motion_enc.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        all_params,
        lr=float(lg_cfg["lr"]),
        weight_decay=float(lg_cfg.get("weight_decay", 1e-2)))

    # ── Scheduler ──
    total_epochs = lg_cfg["epochs"]
    warmup = lg_cfg.get("warmup_epochs", 5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup)
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
        motion_input_type=mi_type)
    val_ds = RadarMotionDataset(
        ds_cfg["val_dir"],
        max_radar_frames=ds_cfg.get("max_seq_len", 100),
        max_motion_frames=ds_cfg.get("max_motion_frames", 300),
        points_per_frame=ds_cfg.get("points_per_frame", 128),
        point_dims=point_dims,
        motion_input_type=mi_type)

    train_loader = DataLoader(
        train_ds, batch_size=lg_cfg["batch_size"],
        shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=lg_cfg["batch_size"],
        shuffle=False, collate_fn=collate_fn, num_workers=2)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val = float("inf")
    history = []
    diag_history = []
    phase_labeler = InteractionPseudoLabeler()
    diag_every = log_cfg.get("diag_every", 5)

    # ── Resume from checkpoint ──
    start_epoch = 0
    if args.resume:
        ckpt_path = args.resume
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if "motion_enc_state_dict" in ckpt:
                motion_enc.load_state_dict(ckpt["motion_enc_state_dict"], strict=False)
            start_epoch = ckpt.get("epoch", 0) + 1
            print(f"  Resumed from {ckpt_path}, starting epoch {start_epoch}")
        else:
            print(f"  ⚠ Resume checkpoint not found: {ckpt_path}")


    # ============================================================
    # Training Loop
    # ============================================================

    for epoch in range(start_epoch, total_epochs):
        model.train()
        motion_enc.train()

        losses = {k: 0.0 for k in
                  ["total", "obs", "kl", "rm_global", "rm_token", "phase", "div"]}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        for batch in pbar:
            pc = batch["point_cloud"].to(device)
            radar_mask = batch["radar_mask"].to(device)
            motion = batch["motion"].to(device)

            # ── Motion encoder → sequence (not mean-pooled) ──
            F_mot = motion_enc(motion)  # (B, T', feat_dim)
            motion_mask_bool = batch["motion_mask"].to(device).bool()
            T_mot = F_mot.shape[1]
            motion_mask_bool = _align_motion_mask(
                motion_mask_bool, T_mot, device)

            # ── Phase pseudo-labels ──
            T_radar = pc.shape[1]
            phase_labels, phase_conf = _generate_batch_phase_labels(
                batch["motion_joints"], T_radar, phase_labeler, device)

            # ── Forward (v7.0: motion_seq + motion_mask) ──
            out = model(pc,
                        motion_seq=F_mot,
                        motion_mask=motion_mask_bool,
                        temporal_mask=radar_mask,
                        phase_labels=phase_labels,
                        phase_confidence=phase_conf)

            # ── NaN check ──
            if torch.isnan(out["loss"]):
                print(f"\n  === NaN at batch {num_batches} ===")
                print(f"    obs={out['loss_obs'].item():.4f} "
                      f"kl={out['loss_kl'].item():.4f} "
                      f"rm_g={out['loss_rm_global'].item():.4f} "
                      f"rm_t={out['loss_rm_token'].item():.4f}")
                continue

            optimizer.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                all_params, lg_cfg.get("grad_clip", 1.0))
            optimizer.step()

            losses["total"] += out["loss"].item()
            losses["obs"] += out["loss_obs"].item()
            losses["kl"] += out["loss_kl"].item()
            losses["rm_global"] += out["loss_rm_global"].item()
            losses["rm_token"] += out["loss_rm_token"].item()
            losses["phase"] += out["loss_phase"].item()
            losses["div"] += out["metric_div"].item()
            num_batches += 1

            if num_batches % log_cfg.get("log_every", 10) == 0:
                pbar.set_postfix(
                    loss=f"{out['loss'].item():.4f}",
                    obs=f"{out['loss_obs'].item():.4f}",
                    rm_g=f"{out['loss_rm_global'].item():.4f}",
                    rm_t=f"{out['loss_rm_token'].item():.4f}",
                    div=f"{out['metric_div'].item():.4f}")

        # ── Scheduler ──
        if epoch < warmup and warmup_scheduler is not None:
            warmup_scheduler.step()
        elif scheduler is not None:
            scheduler.step()

        # ── Validation ──
        model.eval()
        motion_enc.eval()
        val_loss = 0
        val_batches = 0
        last_val_batch = None

        with torch.no_grad():
            for batch in val_loader:
                pc = batch["point_cloud"].to(device)
                mask = batch["radar_mask"].to(device)
                motion = batch["motion"].to(device)

                F_mot = motion_enc(motion)
                motion_mask_bool = batch["motion_mask"].to(device).bool()
                T_mot = F_mot.shape[1]
                motion_mask_bool = _align_motion_mask(
                    motion_mask_bool, T_mot, device)

                T_radar = pc.shape[1]
                phase_labels, phase_conf = _generate_batch_phase_labels(
                    batch["motion_joints"], T_radar, phase_labeler, device)

                out = model(pc,
                            motion_seq=F_mot,
                            motion_mask=motion_mask_bool,
                            temporal_mask=mask,
                            phase_labels=phase_labels,
                            phase_confidence=phase_conf)

                if not torch.isnan(out["loss"]):
                    val_loss += out["loss"].item()
                    val_batches += 1
                last_val_batch = batch

        val_avg = val_loss / max(val_batches, 1)
        n = max(num_batches, 1)
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1}: "
              f"loss={losses['total']/n:.4f} "
              f"(obs={losses['obs']/n:.4f} "
              f"kl={losses['kl']/n:.4f} "
              f"rm_g={losses['rm_global']/n:.4f} "
              f"rm_t={losses['rm_token']/n:.4f} "
              f"phase={losses['phase']/n:.4f}) "
              f"val={val_avg:.4f} lr={lr:.2e} "
              f"| div={losses['div']/n:.4f}")

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": losses["total"] / n,
            "train_obs": losses["obs"] / n,
            "train_kl": losses["kl"] / n,
            "train_rm_global": losses["rm_global"] / n,
            "train_rm_token": losses["rm_token"] / n,
            "train_phase": losses["phase"] / n,
            "val_loss": val_avg,
            "lr": lr,
            "metric_div": losses["div"] / n,
        }

        # ── Diagnostics ──
        if (epoch + 1) % diag_every == 0 and last_val_batch is not None:
            diag_report = run_full_diagnostics(
                model, last_val_batch, device, epoch + 1)
            epoch_record["node_diagnostics"] = diag_report
            diag_history.append({"epoch": epoch + 1, **diag_report})

            cos_sim = diag_report["node_cosine_sim"]
            if cos_sim > 0.9 and epoch > 10:
                print(f"  ⚠ Node collapse persists — "
                      f"consider increasing lambda_token or lambda_diversity")

        history.append(epoch_record)

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

    with open(os.path.join(args.output_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(args.output_dir, "node_diag_history.json"), "w") as f:
        json.dump(diag_history, f, indent=2)

    # ── Final Summary ──
    print(f"\n{'='*60}")
    print(f"Training Complete — v7.0")
    print(f"{'='*60}")
    print(f"  Best val loss: {best_val:.4f}")
    print(f"  Checkpoints: {args.output_dir}")

    if diag_history:
        first = diag_history[0]
        last = diag_history[-1]
        print(f"\n  Node Differentiation Progress:")
        print(f"    cos_sim:  {first['node_cosine_sim']:.4f} → "
              f"{last['node_cosine_sim']:.4f} "
              f"({'↓ improved' if last['node_cosine_sim'] < first['node_cosine_sim'] else '↑ degraded'})")
        print(f"    node_std: {first['node_std']:.4f} → "
              f"{last['node_std']:.4f} "
              f"({'↑ improved' if last['node_std'] > first['node_std'] else '↓ degraded'})")
        print(f"    role_ent: {first['role_entropy_normalized']:.4f} → "
              f"{last['role_entropy_normalized']:.4f} "
              f"({'↑ improved' if last['role_entropy_normalized'] > first['role_entropy_normalized'] else '↓ degraded'})")

        if "token_attn_entropy" in last:
            print(f"    token_attn_entropy: "
                  f"{first.get('token_attn_entropy', 0):.4f} → "
                  f"{last['token_attn_entropy']:.4f}")
            print(f"    token_pattern_sim:  "
                  f"{first.get('token_pattern_similarity', 0):.4f} → "
                  f"{last.get('token_pattern_similarity', 0):.4f} "
                  f"(low=tokens read different nodes)")

        if last['node_cosine_sim'] > 0.8:
            print(f"\n  ⚠ Nodes poorly differentiated "
                  f"(cos_sim={last['node_cosine_sim']:.3f})")
        elif last['node_cosine_sim'] < 0.4:
            print(f"\n  ★ Nodes well differentiated! "
                  f"(cos_sim={last['node_cosine_sim']:.3f})")
        else:
            print(f"\n  ○ Moderate differentiation "
                  f"(cos_sim={last['node_cosine_sim']:.3f})")

    print(f"{'='*60}")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Latent Graph Dynamics v7.0 Training")
    parser.add_argument("--config", default="configs/latent_graph.yaml")
    parser.add_argument("--output_dir", default="checkpoints/latent_graph")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None,
                    help="Path to checkpoint to resume from")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    train(config, args)