#!/usr/bin/env python3
"""
Stage 1 Evaluation — v5.2 compatible.

4 Experiments:
  4.2 Context Interpretability  — c_t가 interaction regime shift를 포착하는지
  4.3 Robustness / Observability — α, Σ가 sensor deprivation을 방어하는지
  4.4 Ablation Studies           — 구조의 당위성
  4.5 Semantic Readiness         — g^radar의 linear probe 성능

사용법:
  python scripts/evaluate.py --config configs/latent_graph.yaml \
      --ckpt checkpoints/latent_graph/latent_graph_best.pt \
      --data_dir data/radar_text_dataset/test

  python scripts/evaluate.py --experiment context --ckpt ...
  python scripts/evaluate.py --experiment robustness --ckpt ...
  python scripts/evaluate.py --experiment ablation --ckpt ...
  python scripts/evaluate.py --experiment linear_probe --ckpt ...
"""

import os, sys, json, argparse, yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.latent_graph_dynamics import LatentGraphDynamicsModel
from models.motion_encoder import build_motion_encoder


# ═══════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════

ACTION_KEYWORDS = {
    "walk": ["walk","walking","walks","stroll"],
    "run": ["run","running","runs","jog","sprint"],
    "sit": ["sit","sitting","sits","sat"],
    "stand": ["stand","standing","stands"],
    "pick_up": ["pick","picks","grab","grabs","grasp","lift"],
    "push": ["push","pushing"],
    "throw": ["throw","throwing","toss"],
    "wave": ["wave","waving"],
    "kick": ["kick","kicking"],
    "punch": ["punch","punching"],
    "jump": ["jump","jumping","hop","leap"],
    "turn": ["turn","turning","rotate","spin"],
    "bend": ["bend","bending","bow","lean"],
    "crouch": ["crouch","squat","kneel"],
}

def text_to_action(text):
    words = set(text.lower().split())
    for action, kws in ACTION_KEYWORDS.items():
        if any(k in words for k in kws):
            return action
    return "other"


class EvalDataset(Dataset):
    def __init__(self, data_dir, max_T=100, max_N=128, D=4,
                 motion_type="humanml3d"):
        self.max_T, self.max_N, self.D = max_T, max_N, D
        self.mi = motion_type
        self.samples = sorted(Path(data_dir).rglob("*.npz"))
        print(f"[EvalDataset] {len(self.samples)} samples, motion={motion_type}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx], allow_pickle=True)
        text = str(data["text"])
        action = str(data["action_label"]) if "action_label" in data \
            else text_to_action(text)

        pc = data["point_cloud"].astype(np.float32)
        Tr = min(pc.shape[0], self.max_T)
        Dc = min(pc.shape[-1], self.D)
        pc_out = np.zeros((self.max_T, self.max_N, self.D), np.float32)
        pc_out[:Tr, :, :Dc] = pc[:Tr, :, :Dc]
        mask = np.zeros(self.max_T, dtype=np.bool_)
        mask[:Tr] = True

        motion = self._load_motion(data)
        return {
            "point_cloud": torch.from_numpy(pc_out),
            "temporal_mask": torch.from_numpy(mask),
            "motion": torch.from_numpy(motion),
            "text": text, "action": action,
            "path": str(self.samples[idx].name),
        }

    def _load_motion(self, data):
        key_map = {"humanml3d": "motion_humanml3d", "latent": "motion_latent",
                    "joints": "motion_joints"}
        dim_map = {"humanml3d": 263, "latent": 201, "joints": 66}
        # 1. Config preferred
        preferred = key_map.get(self.mi, "")
        if preferred and preferred in data:
            m = data[preferred].astype(np.float32)
            return m.reshape(m.shape[0], -1) if m.ndim == 3 else m
        # 2. Fallback
        for k in ["motion_humanml3d", "motion_latent", "motion_joints"]:
            if k in data:
                m = data[k].astype(np.float32)
                return m.reshape(m.shape[0], -1) if m.ndim == 3 else m
        return np.zeros((1, dim_map.get(self.mi, 263)), np.float32)


def collate(batch):
    return {
        "point_cloud": torch.stack([b["point_cloud"] for b in batch]),
        "temporal_mask": torch.stack([b["temporal_mask"] for b in batch]),
        "motion": [b["motion"] for b in batch],
        "texts": [b["text"] for b in batch],
        "actions": [b["action"] for b in batch],
        "paths": [b["path"] for b in batch],
    }


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def load_model(config, ckpt_path, device):
    model = LatentGraphDynamicsModel(config).to(device)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_all(model, loader, device, motion_enc=None):
    """Extract g_radar, g_motion, actions, and per-sample diagnostics."""
    model.eval()
    G_r, G_m, actions, texts = [], [], [], []

    for batch in tqdm(loader, desc="Extracting"):
        pc = batch["point_cloud"].to(device)
        mask = batch["temporal_mask"].to(device)
        g = model.encode(pc, mask)
        G_r.append(g.cpu())
        actions.extend(batch["actions"])
        texts.extend(batch["texts"])

        if motion_enc is not None:
            motions = batch["motion"]
            ml = max(m.shape[0] for m in motions)
            mp = torch.zeros(len(motions), ml, motions[0].shape[-1])
            for i, m in enumerate(motions):
                mp[i, :m.shape[0]] = torch.from_numpy(m) if isinstance(m, np.ndarray) else m
            F_m = motion_enc(mp.to(device))
            G_m.append(F_m.mean(dim=1).cpu())

    result = {"g_radar": torch.cat(G_r), "actions": actions, "texts": texts}
    if G_m:
        result["g_motion"] = torch.cat(G_m)
    return result


# ═══════════════════════════════════════════════════════════
# 4.2 Context Interpretability
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def run_context_interpretability(model, loader, device):
    """
    c_t가 interaction regime shift를 포착하는지 분석.

    1. Action별 c_t 분포 분리도 (inter/intra-class distance ratio)
    2. c_t temporal change rate (||c_t - c_{t-1}||) by action
    3. t-SNE용 c_t 수집 (시각화는 외부에서)
    """
    print("\n" + "=" * 60)
    print("Exp 4.2: Context Interpretability")
    print("=" * 60)

    model.eval()
    # Collect: per-sample average c_t, per-sample c_t trajectory
    action_ctx_means = defaultdict(list)   # action → list of (ctx_dim,)
    action_ctx_change = defaultdict(list)  # action → list of avg ||Δc||
    all_ctx_for_tsne = []                  # list of (ctx_dim,) with labels
    all_ctx_labels = []

    for batch in tqdm(loader, desc="Context analysis"):
        pc = batch["point_cloud"].to(device)
        mask = batch["temporal_mask"].to(device)
        out = model.forward_sequence(pc, mask)
        ctx = out["context_history"]                          # (B, T, ctx_dim)
        B, T, C = ctx.shape

        for b in range(B):
            Tv = int(mask[b].sum().item())
            action = batch["actions"][b]
            ct = ctx[b, :Tv].cpu().numpy()                    # (Tv, C)

            # Mean context
            mean_c = ct.mean(axis=0)
            action_ctx_means[action].append(mean_c)

            # Temporal change rate
            if Tv > 1:
                deltas = np.linalg.norm(ct[1:] - ct[:-1], axis=-1)
                action_ctx_change[action].append(deltas.mean())

            # For t-SNE: sample frames (not all, to manage size)
            step = max(1, Tv // 10)
            for t in range(0, Tv, step):
                all_ctx_for_tsne.append(ct[t])
                all_ctx_labels.append(action)

    # ── Separability: inter/intra class distance ratio ──
    actions_list = sorted(action_ctx_means.keys())
    centroids = {}
    intra_dists = {}
    for a in actions_list:
        vecs = np.stack(action_ctx_means[a])
        centroids[a] = vecs.mean(axis=0)
        intra_dists[a] = np.mean(np.linalg.norm(
            vecs - centroids[a], axis=-1))

    avg_intra = np.mean(list(intra_dists.values()))
    inter_pairs = []
    for i, a1 in enumerate(actions_list):
        for a2 in actions_list[i+1:]:
            inter_pairs.append(np.linalg.norm(
                centroids[a1] - centroids[a2]))
    avg_inter = np.mean(inter_pairs) if inter_pairs else 0
    separation = avg_inter / max(avg_intra, 1e-8)

    print(f"  Context separability: inter/intra = {separation:.4f}")
    print(f"    avg inter-class dist: {avg_inter:.4f}")
    print(f"    avg intra-class dist: {avg_intra:.4f}")

    # ── Temporal change rate by action ──
    print(f"\n  Context temporal change rate (avg ||Δc_t||):")
    change_by_action = {}
    for a in sorted(action_ctx_change.keys()):
        avg = np.mean(action_ctx_change[a])
        change_by_action[a] = float(avg)
        print(f"    {a:15s}: {avg:.4f} (n={len(action_ctx_change[a])})")

    return {
        "separability_ratio": float(separation),
        "avg_inter_dist": float(avg_inter),
        "avg_intra_dist": float(avg_intra),
        "change_rate_by_action": change_by_action,
        "tsne_data": np.stack(all_ctx_for_tsne).tolist(),
        "tsne_labels": all_ctx_labels,
        "n_actions": len(actions_list),
    }


# ═══════════════════════════════════════════════════════════
# 4.3 Robustness / Observability Analysis
# ═══════════════════════════════════════════════════════════

def mask_lower_body_points(pc, point_mask, percentile=30):
    """
    하체 포인트를 강제로 제거.
    z축 기준 하위 percentile의 포인트를 0으로 마스킹.
    """
    pc_c = pc.clone()
    mask_c = point_mask.clone()
    B, T, N, D = pc.shape

    for b in range(B):
        for t in range(T):
            if not point_mask[b, t].any():
                continue
            z_vals = pc[b, t, :, 2]  # z coordinate
            valid = pc[b, t, :, :3].norm(dim=-1) > 1e-6
            if valid.sum() < 2:
                continue
            z_valid = z_vals[valid]
            threshold = torch.quantile(z_valid, percentile / 100.0)
            lower = valid & (z_vals <= threshold)
            pc_c[b, t, lower] = 0
            mask_c[b, t, lower] = False

    return pc_c, mask_c


@torch.no_grad()
def run_robustness(model, loader, device):
    """
    α와 Σ가 sensor deprivation을 어떻게 방어하는지 분석.

    1. Clean vs lower-body-masked: α 변화, Σ 변화
    2. Point drop severity별 reconstruction 성능
    3. Node별 α 반응 (어떤 node가 deprivation에 민감한지)
    """
    print("\n" + "=" * 60)
    print("Exp 4.3: Robustness / Observability Analysis")
    print("=" * 60)

    model.eval()

    # ── Part A: Lower-body masking analysis ──
    print("\n  Part A: Lower-body point masking")
    clean_alpha, masked_alpha = [], []
    clean_sigma, masked_sigma = [], []
    clean_chamfer, masked_chamfer = [], []
    node_alpha_clean = defaultdict(list)
    node_alpha_masked = defaultdict(list)

    for batch in tqdm(loader, desc="Robustness (body mask)"):
        pc = batch["point_cloud"].to(device)
        mask = batch["temporal_mask"].to(device)
        pmask = pc[..., :3].norm(dim=-1) > 1e-6

        # Clean pass
        out_c = model.forward_sequence(pc, mask)

        # Masked pass (lower body removed)
        pc_m, pmask_m = mask_lower_body_points(pc, pmask, percentile=30)
        out_m = model.forward_sequence(pc_m, mask)

        B, T = pc.shape[:2]
        for b in range(B):
            Tv = int(mask[b].sum().item())
            for t in range(Tv):
                # α: confidence per node
                ac = out_c["confidence"][b, t, :, 0].cpu().numpy()
                am = out_m["confidence"][b, t, :, 0].cpu().numpy()
                clean_alpha.append(ac.mean())
                masked_alpha.append(am.mean())

                for m in range(ac.shape[0]):
                    node_alpha_clean[m].append(ac[m])
                    node_alpha_masked[m].append(am[m])

                # Σ: posterior logvar → uncertainty
                sc = out_c["post_logvar"][b, t].exp().mean().item()
                sm = out_m["post_logvar"][b, t].exp().mean().item()
                clean_sigma.append(sc)
                masked_sigma.append(sm)

    ca, ma = np.mean(clean_alpha), np.mean(masked_alpha)
    cs, ms = np.mean(clean_sigma), np.mean(masked_sigma)
    print(f"  Confidence (α):  clean={ca:.4f} → masked={ma:.4f} (Δ={ma-ca:+.4f})")
    print(f"  Uncertainty (Σ): clean={cs:.4f} → masked={ms:.4f} (Δ={ms-cs:+.4f})")

    # Node-level analysis
    print(f"\n  Per-node α change:")
    node_results = {}
    M = len(node_alpha_clean)
    for m in range(M):
        nc = np.mean(node_alpha_clean[m])
        nm = np.mean(node_alpha_masked[m])
        delta = nm - nc
        node_results[m] = {"clean": float(nc), "masked": float(nm),
                           "delta": float(delta)}
        label = "↓ sensitive" if delta < -0.05 else "≈ stable"
        print(f"    Node {m}: {nc:.4f} → {nm:.4f} (Δ={delta:+.4f}) {label}")

    # ── Part B: Point drop severity ──
    print("\n  Part B: Point drop severity curve")
    severities = [0.0, 0.3, 0.5, 0.7, 0.9]
    drop_results = {}

    for sev in severities:
        all_g = []
        all_actions = []
        for batch in loader:
            pc = batch["point_cloud"].to(device)
            mask_t = batch["temporal_mask"].to(device)
            if sev > 0:
                B, T, N, D = pc.shape
                pc_d = pc.clone()
                for b_i in range(B):
                    for t_i in range(T):
                        n_drop = int(N * sev)
                        idx = torch.randperm(N)[:n_drop]
                        pc_d[b_i, t_i, idx] = 0
                g = model.encode(pc_d, mask_t)
            else:
                g = model.encode(pc, mask_t)
            all_g.append(g.cpu())
            all_actions.extend(batch["actions"])

        g_all = torch.cat(all_g).numpy()
        unique = sorted(set(all_actions))
        if len(unique) >= 2:
            amap = {a: i for i, a in enumerate(unique)}
            y = np.array([amap.get(a, 0) for a in all_actions])
            perm = np.random.RandomState(42).permutation(len(y))
            sp = int(len(y) * 0.8)
            knn = KNeighborsClassifier(n_neighbors=min(5, sp-1), metric="cosine")
            knn.fit(g_all[perm[:sp]], y[perm[:sp]])
            acc = float(accuracy_score(y[perm[sp:]], knn.predict(g_all[perm[sp:]])))
        else:
            acc = 0.0

        drop_results[sev] = acc
        print(f"    drop={sev:.0%}: kNN_acc={acc:.4f}")

    return {
        "body_mask": {
            "alpha_clean": float(ca), "alpha_masked": float(ma),
            "sigma_clean": float(cs), "sigma_masked": float(ms),
            "per_node": node_results,
        },
        "point_drop_curve": drop_results,
    }


# ═══════════════════════════════════════════════════════════
# 4.4 Ablation Studies
# ═══════════════════════════════════════════════════════════

ABLATION_CONFIGS = {
    "w/o_context": {"disable_context": True},
    "w/o_self_attn": {"disable_self_attn": True},
    "w/o_confidence": {"disable_confidence": True},
    "w/o_contrastive": {"lambda_motion": 0.0},
}


def create_ablation_config(base_config, ablation_name):
    import copy
    config = copy.deepcopy(base_config)
    lg = config["latent_graph"]
    overrides = ABLATION_CONFIGS[ablation_name]
    for k, v in overrides.items():
        lg[k] = v
    return config


@torch.no_grad()
def run_ablation(base_ckpt, ablation_dir, loader, device, config):
    """
    w/o Context:     c_t 없이 고정 transition
    w/o Self-Attn:   implicit relation 제거
    w/o Confidence:  α=1 고정 (항상 evidence 신뢰)
    w/o Contrastive: motion alignment 제거
    """
    print("\n" + "=" * 60)
    print("Exp 4.4: Ablation Studies")
    print("=" * 60)

    results = {}

    # Full model
    model = load_model(config, base_ckpt, device)
    emb = extract_all(model, loader, device)
    full_acc = _quick_knn(emb)
    full_chamfer = _quick_chamfer(model, loader, device)
    results["full"] = {"knn_acc": full_acc, "chamfer": full_chamfer}
    print(f"  Full model: kNN={full_acc:.4f}, Chamfer={full_chamfer:.6f}")

    # Ablations
    for abl_name in ABLATION_CONFIGS:
        ckpt = os.path.join(ablation_dir, f"{abl_name}_best.pt")
        if not os.path.exists(ckpt):
            print(f"  {abl_name:20s}: ✗ not found")
            results[abl_name] = {"status": "missing"}
            continue

        abl_cfg = create_ablation_config(config, abl_name)
        try:
            abl_model = load_model(abl_cfg, ckpt, device)
            abl_emb = extract_all(abl_model, loader, device)
            acc = _quick_knn(abl_emb)
            chamfer = _quick_chamfer(abl_model, loader, device)
            da, dc = acc - full_acc, chamfer - full_chamfer
            results[abl_name] = {"knn_acc": acc, "chamfer": chamfer}
            print(f"  {abl_name:20s}: kNN={acc:.4f} (Δ={da:+.4f}), "
                  f"Chamfer={chamfer:.6f} (Δ={dc:+.6f})")
        except Exception as e:
            print(f"  {abl_name:20s}: ✗ {e}")
            results[abl_name] = {"status": "error", "error": str(e)}

    return results


def _quick_knn(emb):
    g = emb["g_radar"].numpy()
    actions = emb["actions"]
    unique = sorted(set(actions))
    if len(unique) < 2:
        return 0.0
    amap = {a: i for i, a in enumerate(unique)}
    y = np.array([amap[a] for a in actions])
    perm = np.random.RandomState(42).permutation(len(y))
    sp = int(len(y) * 0.8)
    knn = KNeighborsClassifier(n_neighbors=min(5, sp-1), metric="cosine")
    knn.fit(g[perm[:sp]], y[perm[:sp]])
    return float(accuracy_score(y[perm[sp:]], knn.predict(g[perm[sp:]])))


@torch.no_grad()
def _quick_chamfer(model, loader, device, max_batches=20):
    model.eval()
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        pc = batch["point_cloud"].to(device)
        mask = batch["temporal_mask"].to(device)
        out = model.forward_sequence(pc, mask)
        recon = out["recon_sequence"]
        gt = pc[..., :3]
        gt_valid = gt.norm(dim=-1) > 1e-6
        B, T = pc.shape[:2]
        for b in range(B):
            Tv = int(mask[b].sum().item())
            for t in range(min(Tv, 5)):  # sample frames
                vm = gt_valid[b, t]
                if vm.sum() < 2:
                    continue
                g, p = gt[b, t, vm], recon[b, t]
                gn = g - g.mean(0, keepdim=True)
                pn = p - p.mean(0, keepdim=True)
                d = (pn.unsqueeze(1) - gn.unsqueeze(0)).pow(2).sum(-1)
                total += (d.min(1).values.mean() + d.min(0).values.mean()).item()
                count += 1
    return total / max(count, 1)


# ═══════════════════════════════════════════════════════════
# 4.5 Semantic Readiness (Linear Probing)
# ═══════════════════════════════════════════════════════════

def run_linear_probe(embeddings, config):
    """
    g^radar 위에 linear layer 하나로 분류.
    높은 정확도 = representation이 semantic하게 잘 정렬됨.
    """
    print("\n" + "=" * 60)
    print("Exp 4.5: Semantic Readiness (Linear Probing)")
    print("=" * 60)

    g = embeddings["g_radar"].numpy()
    actions = embeddings["actions"]
    unique = sorted(set(actions))

    if len(unique) < 2:
        print("  ✗ < 2 classes")
        return {"status": "skipped"}

    amap = {a: i for i, a in enumerate(unique)}
    y = np.array([amap[a] for a in actions])
    K = len(unique)

    print(f"  Classes ({K}): {unique}")
    dist = dict(zip(*np.unique(y, return_counts=True)))
    print(f"  Distribution: {dist}")

    N = len(y)
    perm = np.random.RandomState(42).permutation(N)
    sp = int(N * 0.8)
    tr_idx, te_idx = perm[:sp], perm[sp:]

    results = {}

    # ── k-NN ──
    knn = KNeighborsClassifier(n_neighbors=min(5, sp-1), metric="cosine")
    knn.fit(g[tr_idx], y[tr_idx])
    knn_pred = knn.predict(g[te_idx])
    knn_acc = accuracy_score(y[te_idx], knn_pred)
    knn_f1 = f1_score(y[te_idx], knn_pred, average="weighted", zero_division=0)
    print(f"  k-NN (k=5): acc={knn_acc:.4f}, F1={knn_f1:.4f}")
    results["knn_accuracy"] = knn_acc
    results["knn_f1"] = knn_f1

    # ── Linear probe ──
    D = g.shape[1]
    probe = nn.Linear(D, K)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    Xt = torch.from_numpy(g[tr_idx]).float()
    yt = torch.from_numpy(y[tr_idx]).long()
    Xe = torch.from_numpy(g[te_idx]).float()

    for ep in range(50):
        probe.train()
        loss = F.cross_entropy(probe(Xt), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        pred = probe(Xe).argmax(1).numpy()
    lin_acc = accuracy_score(y[te_idx], pred)
    lin_f1 = f1_score(y[te_idx], pred, average="weighted", zero_division=0)
    print(f"  Linear (50ep): acc={lin_acc:.4f}, F1={lin_f1:.4f}")
    results["linear_accuracy"] = lin_acc
    results["linear_f1"] = lin_f1

    # Confusion matrix
    results["confusion_matrix"] = confusion_matrix(y[te_idx], pred).tolist()
    results["class_names"] = unique
    results["random_baseline"] = 1.0 / K
    results["num_classes"] = K

    # ── Cross-modal retrieval (if motion available) ──
    if "g_motion" in embeddings:
        print("\n  Cross-modal Retrieval:")
        gr = F.normalize(embeddings["g_radar"], dim=-1).numpy()
        gm = F.normalize(embeddings["g_motion"], dim=-1).numpy()
        N_all = gr.shape[0]

        for name, q, gal in [("R→M", gr, gm), ("M→R", gm, gr)]:
            sim = q @ gal.T
            ranks = np.array([np.where(np.argsort(-sim[i]) == i)[0][0] + 1
                              for i in range(N_all)])
            r1 = (ranks <= 1).mean()
            r5 = (ranks <= 5).mean()
            mrr = (1.0 / ranks).mean()
            print(f"    {name}: R@1={r1:.4f}, R@5={r5:.4f}, MRR={mrr:.4f}")
            results[f"retrieval_{name}"] = {"R@1": float(r1), "R@5": float(r5),
                                            "MRR": float(mrr)}

    return results


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser("Stage 1 Evaluation (v5.2)")
    p.add_argument("--config", default="configs/latent_graph.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--experiment", default="all",
                   choices=["all","context","robustness","ablation",
                            "linear_probe"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--output", default="results/stage1_eval.json")
    p.add_argument("--ablation_dir", default="checkpoints/ablations")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(config, args.ckpt, device)

    motion_enc = None
    try:
        motion_enc = build_motion_encoder(config).to(device)
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        if "motion_enc_state_dict" in ckpt:
            motion_enc.load_state_dict(ckpt["motion_enc_state_dict"], strict=False)
        motion_enc.eval()
        for p in motion_enc.parameters():
            p.requires_grad = False
    except Exception as e:
        print(f"Motion encoder not available: {e}")

    lg = config["latent_graph"]
    me = config.get("motion_encoder", {})
    ds = EvalDataset(
        args.data_dir,
        max_T=config.get("dataset", {}).get("max_seq_len", 100),
        max_N=config.get("dataset", {}).get("points_per_frame", 128),
        D=lg.get("point_in_dim", 4),
        motion_type=me.get("input_type", "humanml3d"))
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate, num_workers=2)

    R = {"config": args.config, "ckpt": args.ckpt}
    run_all = args.experiment == "all"

    # Shared embeddings for linear probe and retrieval
    emb = None
    if run_all or args.experiment == "linear_probe":
        emb = extract_all(model, loader, device, motion_enc)

    if run_all or args.experiment == "context":
        R["context"] = run_context_interpretability(model, loader, device)

    if run_all or args.experiment == "robustness":
        R["robustness"] = run_robustness(model, loader, device)

    if run_all or args.experiment == "ablation":
        R["ablation"] = run_ablation(
            args.ckpt, args.ablation_dir, loader, device, config)

    if run_all or args.experiment == "linear_probe":
        R["linear_probe"] = run_linear_probe(emb, config)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    class Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return super().default(o)

    with open(args.output, "w") as f:
        json.dump(R, f, indent=2, ensure_ascii=False, cls=Enc)
    print(f"\nSaved: {args.output}")

    # Summary
    print(f"\n{'='*60}\nSummary\n{'='*60}")
    if "context" in R:
        print(f"  Context separability: {R['context']['separability_ratio']:.4f}")
    if "robustness" in R:
        bm = R["robustness"]["body_mask"]
        print(f"  α drop (body mask): {bm['alpha_clean']:.4f} → {bm['alpha_masked']:.4f}")
        print(f"  Σ rise (body mask): {bm['sigma_clean']:.4f} → {bm['sigma_masked']:.4f}")
    if "linear_probe" in R and "linear_accuracy" in R["linear_probe"]:
        lp = R["linear_probe"]
        print(f"  Linear probe: {lp['linear_accuracy']:.4f}")
        print(f"  k-NN:         {lp['knn_accuracy']:.4f}")


if __name__ == "__main__":
    main()