#!/usr/bin/env python3
"""
Latent Graph Dynamics — v6.0 Training Script

v5.x → v6.0 변경사항:
  - NodeDiagnostics 통합: 매 epoch 노드 분화 상태 자동 체크
  - metric_div 진단 로깅 추가 (loss가 아닌 모니터링 지표)
  - 학습 초반 node collapse 감지 시 경고
  - train_history에 node diagnostics 포함

사용법:
  python scripts/train_latent_graph_v6.py --config configs/latent_graph.yaml
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

# ★ v6 모델 import (기존 모델 대신)
from models.latent_graph_dynamics import (
    LatentGraphDynamicsModel,
    InteractionPseudoLabeler,
    NodeDiagnostics,
)
from models.motion_encoder import build_motion_encoder


# ============================================================
# Dataset (기존과 동일)
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
        motion_padded = np.zeros((self.max_motion_frames, motion_dim), dtype=np.float32)
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


# ============================================================
# ★ Enhanced Diagnostics Logger
# ============================================================

@torch.no_grad()
def run_full_diagnostics(model, batch, device, epoch, step=0):
    """
    매 epoch 끝에 한 배치로 전체 진단 수행.
    노드 분화 상태 + 추가 통계.
    """
    model.eval()
    pc = batch["point_cloud"].to(device)
    mask = batch["radar_mask"].to(device)

    out = model.forward_sequence(pc, mask)

    # ── NodeDiagnostics (기본) ──
    node_report = model.get_diagnostics(out, mask)
    NodeDiagnostics.log(node_report, epoch, step)

    # ── 추가 통계 ──
    node_seq = out["node_history"]       # (B, T, M, D)
    conf_seq = out["confidence"]         # (B, T, M, 1)
    B, T, M, D = node_seq.shape

    # Query diversity (학습된 node_queries 간 유사도)
    queries = model.node_queries.squeeze(0)  # (M, D)
    q_norm = F.normalize(queries, dim=-1)
    q_sim = (q_norm @ q_norm.t())
    q_eye = torch.eye(M, device=q_sim.device)
    q_off = (q_sim * (1 - q_eye)).abs().mean()
    node_report["query_cosine_sim"] = float(q_off.item())

    # Per-node mean confidence
    conf_per_node = conf_seq.squeeze(-1).mean(dim=(0, 1))  # (M,)
    node_report["per_node_confidence"] = conf_per_node.tolist()

    # Temporal consistency: node 간 role이 시간에 따라 유지되는지
    # 각 프레임에서 가장 높은 confidence의 node index
    argmax_nodes = conf_seq.squeeze(-1).argmax(dim=2)  # (B, T)
    # 연속 프레임 간 같은 node가 선택되는 비율
    if T > 1:
        same_node = (argmax_nodes[:, 1:] == argmax_nodes[:, :-1]).float().mean()
    else:
        same_node = 1.0
    node_report["temporal_role_consistency"] = float(same_node.item() if isinstance(same_node, torch.Tensor) else same_node)

    # FiLM diversity: transition의 gamma가 node별로 얼마나 다른지
    # (이미 forward 끝났으므로 context_history에서 간접 확인)
    ctx = out["context_history"]  # (B, T, ctx_dim)
    ctx_std = ctx.std(dim=1).mean()
    node_report["context_temporal_std"] = float(ctx_std.item())

    # ── 상세 출력 ──
    print(f"  [NodeDiag Extended] ep{epoch}:")
    print(f"    query_cos_sim={q_off:.4f} (low=good)")
    print(f"    per_node_conf={[f'{c:.3f}' for c in conf_per_node.tolist()]}")
    print(f"    temporal_role_consistency={node_report['temporal_role_consistency']:.3f} "
          f"(1.0=always same node wins, low=dynamic)")
    print(f"    context_temporal_std={ctx_std:.4f}")

    # ── 경고 ──
    if node_report["node_cosine_sim"] > 0.85 and epoch >= 5:
        print(f"  ⚠⚠⚠ WARNING: Node collapse detected at epoch {epoch}! "
              f"cos_sim={node_report['node_cosine_sim']:.4f}")
        print(f"       Consider: increase init_rounds, reduce beta_kl, or check L_obs trend")

    model.train()
    return node_report


# ============================================================
# Training Loop
# ============================================================

def train(config, args):
    print("\n" + "=" * 60)
    print("Stage 1: Latent Graph Dynamics Training — v6.0")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lg_cfg = config["latent_graph"]
    ds_cfg = config.get("dataset", {})
    me_cfg = config.get("motion_encoder", {})
    log_cfg = config.get("logging", {})

    # ── Model ──
    model = LatentGraphDynamicsModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Latent Graph Model v6.0: {total_params:,} params")
    print(f"  Nodes: {model.num_nodes}, dim: {model.node_dim}")
    print(f"  Context: {model.ctx_dim}D, history K={model.ctx_history_len}")
    print(f"  Output: {model.out_dim}D")
    print(f"  Queue size: {model.queue_size}")
    print(f"  Loss: L_obs + {model.beta_kl}·L_KL + {model.lambda_motion}·L_rm + {model.lambda_diversity}·L_div")

    # ── 초기 node query 상태 확인 ──
    queries = model.node_queries.squeeze(0)
    q_norm = F.normalize(queries, dim=-1)
    q_sim = (q_norm @ q_norm.t())
    q_eye = torch.eye(model.num_nodes, device=q_sim.device)
    init_sim = (q_sim * (1 - q_eye)).abs().mean()
    print(f"  Initial query cos_sim: {init_sim:.4f} "
          f"(orthogonal init → should be ~0)")

    # ── Motion Encoder ──
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
        shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=lg_cfg["batch_size"],
        shuffle=False, collate_fn=collate_fn, num_workers=2)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val = float("inf")
    history = []
    diag_history = []
    phase_labeler = InteractionPseudoLabeler()
    diag_every = log_cfg.get("diag_every", 5)  # 진단 주기

    # ============================================================
    # Training Loop
    # ============================================================

    for epoch in range(total_epochs):
        model.train()
        motion_enc.train()

        loss_keys = ["total", "obs", "kl", "rm", "phase"]
        metric_keys = ["metric_div"]  # diagnostic only, not in loss
        losses = {k: 0.0 for k in loss_keys}
        metrics = {k: 0.0 for k in metric_keys}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        for batch in pbar:
            pc = batch["point_cloud"].to(device)
            radar_mask = batch["radar_mask"].to(device)
            motion = batch["motion"].to(device)

            F_mot = motion_enc(motion)
            m_mask = batch["motion_mask"].to(device).float()
            T_out = F_mot.shape[1]
            m_mask = F.adaptive_max_pool1d(
                m_mask.unsqueeze(1), T_out).squeeze(1)
            g_motion = (F_mot * m_mask.unsqueeze(-1)).sum(dim=1) / \
                       m_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

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
                key = f"loss_{k}"
                if key in out:
                    losses[k] += out[key].item()
            # Diagnostic metric (not in loss)
            if "metric_div" in out:
                metrics["metric_div"] += out["metric_div"].item()
            num_batches += 1

            if num_batches % log_cfg.get("log_every", 10) == 0:
                pbar.set_postfix(
                    loss=f"{out['loss'].item():.4f}",
                    obs=f"{out['loss_obs'].item():.4f}",
                    div=f"{out['metric_div'].item():.4f}",  # diagnostic
                )

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
                m_mask = batch["motion_mask"].to(device).float()
                T_out = F_mot.shape[1]
                m_mask = F.adaptive_max_pool1d(
                    m_mask.unsqueeze(1), T_out).squeeze(1)
                g_motion = (F_mot * m_mask.unsqueeze(-1)).sum(dim=1) / \
                           m_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

                T_radar = pc.shape[1]
                phase_labels, phase_conf = _generate_batch_phase_labels(
                    batch["motion_joints"], T_radar, phase_labeler, device)

                out = model(pc, motion_features=g_motion,
                            temporal_mask=mask,
                            phase_labels=phase_labels,
                            phase_confidence=phase_conf)
                val_loss += out["loss"].item()
                val_batches += 1
                last_val_batch = batch  # 진단용

        val_avg = val_loss / max(val_batches, 1)
        n = max(num_batches, 1)
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1}: "
              f"loss={losses['total']/n:.4f} "
              f"(obs={losses['obs']/n:.4f} "
              f"kl={losses['kl']/n:.4f} "
              f"rm={losses['rm']/n:.4f} "
              f"phase={losses['phase']/n:.4f}) "
              f"val={val_avg:.4f} lr={lr:.2e} "
              f"| div_metric={metrics['metric_div']/n:.4f}")

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": losses["total"] / n,
            "train_obs": losses["obs"] / n,
            "train_kl": losses["kl"] / n,
            "train_rm": losses["rm"] / n,
            "train_phase": losses["phase"] / n,
            "val_loss": val_avg,
            "lr": lr,
            "metric_div": metrics["metric_div"] / n,  # diagnostic
        }

        # ════════════════════════════════════════════════
        # ★ Node Differentiation Diagnostics
        # ════════════════════════════════════════════════
        if (epoch + 1) % diag_every == 0 and last_val_batch is not None:
            diag_report = run_full_diagnostics(
                model, last_val_batch, device, epoch + 1)
            epoch_record["node_diagnostics"] = diag_report
            diag_history.append({
                "epoch": epoch + 1,
                **diag_report,
            })

            # ── 학습 상태 판단 ──
            cos_sim = diag_report["node_cosine_sim"]
            if cos_sim > 0.9 and epoch > 10:
                print(f"  ⚠ Node collapse persists — structural fixes may need tuning")
                print(f"    Try: increase init_rounds, or reduce beta_kl")

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

    # ── Node diagnostics history ──
    with open(os.path.join(args.output_dir, "node_diag_history.json"), "w") as f:
        json.dump(diag_history, f, indent=2)

    # ════════════════════════════════════════════════
    # ★ Final Summary with Node Health Check
    # ════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Training Complete — v6.0")
    print(f"{'='*60}")
    print(f"  Best val loss: {best_val:.4f}")
    print(f"  Checkpoints: {args.output_dir}")

    if diag_history:
        first_diag = diag_history[0]
        last_diag = diag_history[-1]
        print(f"\n  Node Differentiation Progress:")
        print(f"    cos_sim:  {first_diag['node_cosine_sim']:.4f} → "
              f"{last_diag['node_cosine_sim']:.4f} "
              f"({'↓ improved' if last_diag['node_cosine_sim'] < first_diag['node_cosine_sim'] else '↑ degraded'})")
        print(f"    node_std: {first_diag['node_std']:.4f} → "
              f"{last_diag['node_std']:.4f} "
              f"({'↑ improved' if last_diag['node_std'] > first_diag['node_std'] else '↓ degraded'})")
        print(f"    role_ent: {first_diag['role_entropy_normalized']:.4f} → "
              f"{last_diag['role_entropy_normalized']:.4f} "
              f"({'↑ improved' if last_diag['role_entropy_normalized'] > first_diag['role_entropy_normalized'] else '↓ degraded'})")

        if last_diag['node_cosine_sim'] > 0.8:
            print(f"\n  ⚠ Nodes are still poorly differentiated (cos_sim={last_diag['node_cosine_sim']:.3f})")
            print(f"    Suggestions:")
            print(f"    - Increase init_rounds (currently {config['latent_graph'].get('init_rounds', 3)})")
            print(f"    - Reduce beta_kl (currently {model.beta_kl})")
            print(f"    - Check if L_obs is decreasing (reconstruction pressure drives specialization)")
        elif last_diag['node_cosine_sim'] < 0.4:
            print(f"\n  ★ Nodes are well differentiated! (cos_sim={last_diag['node_cosine_sim']:.3f})")
        else:
            print(f"\n  ○ Moderate node differentiation (cos_sim={last_diag['node_cosine_sim']:.3f})")

    print(f"{'='*60}")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Latent Graph Dynamics v6.0 Training")
    parser.add_argument("--config", default="configs/latent_graph.yaml")
    parser.add_argument("--output_dir", default="checkpoints/latent_graph")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    train(config, args)