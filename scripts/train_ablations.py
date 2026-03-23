#!/usr/bin/env python3
"""
Ablation 모델 학습 스크립트

Full model에서 한 가지씩 컴포넌트를 비활성화하고 동일 조건으로 재학습.
evaluate.py --experiment ablation 이 학습된 checkpoint을 필요로 함.

사용법:
  # 모든 ablation 학습
  python scripts/train_ablations.py --config configs/latent_graph.yaml

  # 특정 ablation만
  python scripts/train_ablations.py --config configs/latent_graph.yaml \
      --ablation no_obs_gate
"""

import os
import sys
import copy
import argparse
import yaml
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


ABLATION_LIST = [
    "no_obs_gate",       # observability gate 비활성 (항상 o=1)
    "no_relation",       # relation module 비활성 (R_t=0)
    "no_gru",            # GRU 비활성 (simple linear transition)
    "no_kl",             # L_KL 제거 (beta=0)
    "no_contrastive",    # L_(r-m) 제거 (lambda=0)
]


def patch_model_for_ablation(model, ablation_name):
    """
    학습된 모델이 아닌, 모델 구조를 직접 패치하여 컴포넌트 비활성화.
    config으로 못 하는 것은 forward를 monkey-patch.
    """
    if ablation_name == "no_obs_gate":
        # Gate를 항상 1로 고정
        original_forward = model.obs_gate.forward
        def always_one(prior, evidence):
            return torch.ones(prior.shape[0], prior.shape[1], 1,
                              device=prior.device)
        model.obs_gate.forward = always_one

    elif ablation_name == "no_relation":
        # Relation을 항상 0으로
        original_forward = model.relation.forward
        def zero_relation(nodes, prev_relations=None):
            B, M, D = nodes.shape
            return torch.zeros(B, M, M, model.rel_dim, device=nodes.device)
        model.relation.forward = zero_relation

    elif ablation_name == "no_gru":
        # GRU를 단순 linear로 대체
        original_forward = model.transition.gru
        model.transition.gru = torch.nn.Linear(
            model.node_dim, model.node_dim).to(next(model.parameters()).device)

    return model


def make_ablation_config(base_config, ablation_name):
    config = copy.deepcopy(base_config)
    lg = config["latent_graph"]

    if ablation_name == "no_kl":
        lg["beta_kl"] = 0.0
    elif ablation_name == "no_contrastive":
        lg["lambda_motion"] = 0.0

    # 학습 시간 줄이기: full model의 절반
    lg["epochs"] = max(lg.get("epochs", 100) // 2, 20)

    return config


def train_single_ablation(base_config, ablation_name, output_dir, args):
    """단일 ablation 학습."""
    print(f"\n{'='*60}")
    print(f"Ablation: {ablation_name}")
    print(f"{'='*60}")

    config = make_ablation_config(base_config, ablation_name)

    # train_latent_graph.py의 train() 함수를 재사용
    from scripts.train_latent_graph import train

    abl_output = os.path.join(output_dir, ablation_name)
    os.makedirs(abl_output, exist_ok=True)

    # args 복사 + output_dir 변경
    abl_args = copy.copy(args)
    abl_args.output_dir = abl_output

    # 구조적 ablation은 모델 생성 후 패치 필요
    if ablation_name in ["no_obs_gate", "no_relation", "no_gru"]:
        # 일단 학습 시작하되, 모델 생성 직후 패치
        # → train() 함수를 직접 수정하기보다, 여기서 전체 루프 실행
        _train_with_patch(config, ablation_name, abl_args)
    else:
        # loss weight 변경만으로 충분한 ablation
        train(config, abl_args)

    # Best checkpoint 이름 통일
    best_src = os.path.join(abl_output, "latent_graph_best.pt")
    best_dst = os.path.join(output_dir, f"{ablation_name}_best.pt")
    if os.path.exists(best_src):
        import shutil
        shutil.copy2(best_src, best_dst)
        print(f"  Saved: {best_dst}")


def _train_with_patch(config, ablation_name, args):
    """구조적 ablation: 모델 패치 후 학습."""
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from scripts.train_latent_graph import (
        RadarMotionDataset, collate_fn, log_diagnostics
    )
    from models.latent_graph_dynamics import LatentGraphDynamicsModel
    from models.motion_encoder import build_motion_encoder

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lg_cfg = config["latent_graph"]
    ds_cfg = config.get("dataset", {})
    me_cfg = config.get("motion_encoder", {})

    model = LatentGraphDynamicsModel(config).to(device)
    model = patch_model_for_ablation(model, ablation_name)

    motion_enc = build_motion_encoder(config).to(device)

    all_params = list(model.parameters())
    all_params += [p for p in motion_enc.parameters() if p.requires_grad]
    optimizer = optim.AdamW(all_params, lr=float(lg_cfg["lr"]))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=lg_cfg["epochs"])

    mi_type = me_cfg.get("input_type", "humanml3d")
    train_ds = RadarMotionDataset(
        ds_cfg["train_dir"], point_dims=lg_cfg.get("point_in_dim", 4),
        motion_input_type=mi_type)
    val_ds = RadarMotionDataset(
        ds_cfg["val_dir"], point_dims=lg_cfg.get("point_in_dim", 4),
        motion_input_type=mi_type)

    train_loader = DataLoader(train_ds, batch_size=lg_cfg["batch_size"],
                              shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=lg_cfg["batch_size"],
                            shuffle=False, collate_fn=collate_fn, num_workers=2)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val = float("inf")

    from tqdm import tqdm

    for epoch in range(lg_cfg["epochs"]):
        model.train()
        motion_enc.train()
        total_loss = 0
        n = 0

        for batch in tqdm(train_loader,
                          desc=f"[{ablation_name}] Ep {epoch+1}/{lg_cfg['epochs']}"):
            pc = batch["point_cloud"].to(device)
            mask = batch["radar_mask"].to(device)
            motion = batch["motion"].to(device)

            F_mot = motion_enc(motion)
            g_motion = F_mot.mean(dim=1)
            out = model(pc, motion_features=g_motion, temporal_mask=mask)

            optimizer.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            total_loss += out["loss"].item()
            n += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                pc = batch["point_cloud"].to(device)
                mask = batch["radar_mask"].to(device)
                motion = batch["motion"].to(device)
                F_mot = motion_enc(motion)
                g_motion = F_mot.mean(dim=1)
                out = model(pc, motion_features=g_motion, temporal_mask=mask)
                val_loss += out["loss"].item()
                vn += 1

        val_avg = val_loss / max(vn, 1)
        print(f"  Epoch {epoch+1}: train={total_loss/max(n,1):.4f}, "
              f"val={val_avg:.4f}")

        if val_avg < best_val:
            best_val = val_avg
            torch.save({
                "model_state_dict": model.state_dict(),
                "motion_enc_state_dict": motion_enc.state_dict(),
                "ablation": ablation_name,
                "val_loss": val_avg,
            }, os.path.join(args.output_dir, "latent_graph_best.pt"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/latent_graph.yaml")
    parser.add_argument("--output_dir", default="checkpoints/ablations")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ablation", default=None,
                        help="단일 ablation만 실행 (None=전체)")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.ablation:
        train_single_ablation(config, args.ablation, args.output_dir, args)
    else:
        for abl in ABLATION_LIST:
            train_single_ablation(config, abl, args.output_dir, args)

    print(f"\nAll ablations saved to {args.output_dir}")


if __name__ == "__main__":
    main()