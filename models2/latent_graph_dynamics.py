"""
Latent Graph Dynamics — v8.0 (HY-Motion 201D Direct Prediction)

v7.0 → v8.0 changes:
  [removed] MotionSemanticHead — 더 이상 motion sequence에서 token 추출 안 함
  [removed] radar_proj, motion_proj — contrastive projection heads 제거
  [removed] motion_queue — momentum queue 제거
  [removed] global_contrastive_loss, token_contrastive_loss — contrastive 제거
  [added]   hymotion_head — g_radar(512D) → 201D 직접 예측
  [changed] forward() — motion_latent_201 직접 받아서 MSE loss

핵심 아이디어:
  HY-Motion의 201D latent space는 이미 text-conditioned flow matching으로
  학습된 "완성된" 공간. 이 공간을 변형하지 않고 직접 target으로 사용.
  
  201D에 정렬되면 text 정렬은 자동으로 따라옴 (HY-Motion이 이미 학습).
  추론 시 pred_201D → HY-Motion decode_motion_from_latent()로 full motion 복원 가능.

Loss: L_obs + β·L_KL + λ_latent·MSE(pred_201, gt_201) [+ λ_ph·L_phase] [+ λ_d·L_div]

기존 유지:
  [fix-1] Node-specific FiLM in transition
  [fix-2] Orthogonal node query init
  [fix-3] Iterative slot competition init
  [fix-A] Node query bias in EvidenceExtractor
  [fix-7] NodeDiagnostics
  NodeAwareReadout (semantic tokens + g_radar 생성은 그대로)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


# ═══════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════

def reparameterize(mu, logvar, training=True):
    if not training:
        return mu
    return mu + (0.5 * logvar).exp() * torch.randn_like(mu)


def kl_divergence_per_sample(mu_q, lv_q, mu_p, lv_p):
    var_q = lv_q.exp()
    var_p = lv_p.exp().clamp(min=1e-8)
    kl = 0.5 * (lv_p - lv_q + var_q / var_p
                 + (mu_q - mu_p).pow(2) / var_p - 1)
    return kl.mean(dim=list(range(1, kl.dim())))


# ═══════════════════════════════════════════════════════════
# NodeDiagnostics
# ═══════════════════════════════════════════════════════════

class NodeDiagnostics:
    @staticmethod
    @torch.no_grad()
    def compute(model_output: dict, temporal_mask=None) -> dict:
        node_seq = model_output["node_history"]
        conf_seq = model_output["confidence"]
        B, T, M, D = node_seq.shape

        node_frame = node_seq.mean(dim=1)
        node_norm = F.normalize(node_frame, dim=-1)
        sim_matrix = torch.bmm(node_norm, node_norm.transpose(1, 2))
        eye = torch.eye(M, device=sim_matrix.device).unsqueeze(0)
        off_diag = sim_matrix * (1 - eye)
        node_cos_sim = off_diag.sum() / (B * M * (M - 1))

        node_std = node_frame.std(dim=1).mean()

        conf_per_frame = conf_seq.squeeze(-1)
        if temporal_mask is not None:
            valid = temporal_mask.unsqueeze(-1).float()
            conf_std = (conf_per_frame.std(dim=2) * valid.squeeze(-1)).sum() / valid.sum().clamp(1)
        else:
            conf_std = conf_per_frame.std(dim=2).mean()

        argmax_nodes = conf_per_frame.argmax(dim=2)
        node_counts = torch.zeros(B, M, device=node_seq.device)
        for m in range(M):
            node_counts[:, m] = (argmax_nodes == m).float().sum(dim=1)
        node_probs = node_counts / node_counts.sum(dim=1, keepdim=True).clamp(1)
        role_entropy = -(node_probs * (node_probs + 1e-10).log()).sum(dim=1).mean()
        max_entropy = np.log(M)

        return {
            "node_cosine_sim": float(node_cos_sim.item()),
            "node_std": float(node_std.item()),
            "conf_std_across_nodes": float(conf_std.item()),
            "role_entropy": float(role_entropy.item()),
            "role_entropy_normalized": float(role_entropy.item() / max_entropy),
            "max_entropy": float(max_entropy),
        }

    @staticmethod
    def log(report: dict, epoch: int, step: int = 0, prefix: str = "  [NodeDiag]"):
        cos = report["node_cosine_sim"]
        std = report["node_std"]
        cstd = report["conf_std_across_nodes"]
        ent = report["role_entropy_normalized"]
        if cos > 0.9:
            status = "⚠ COLLAPSED"
        elif cos > 0.7:
            status = "△ WEAK"
        elif cos > 0.4:
            status = "○ MODERATE"
        else:
            status = "★ GOOD"
        print(f"{prefix} ep{epoch} step{step}: "
              f"cos_sim={cos:.4f} node_std={std:.4f} "
              f"conf_std={cstd:.4f} role_ent={ent:.2f} → {status}")


# ═══════════════════════════════════════════════════════════
# Block 1. Observation Encoding
# ═══════════════════════════════════════════════════════════

class ObservationEncoder(nn.Module):
    def __init__(self, in_dim=4, hidden_dims=None, out_dim=256):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128]
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU()])
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.mlp = nn.Sequential(*layers)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, Y):
        return self.out_norm(self.mlp(Y))


# ═══════════════════════════════════════════════════════════
# Block 2. Latent Node Initialization — Slot Competition
# ═══════════════════════════════════════════════════════════

class LatentNodeInitializer(nn.Module):
    def __init__(self, node_dim, feat_dim, num_heads=4, num_rounds=3):
        super().__init__()
        self.num_rounds = num_rounds
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=node_dim, num_heads=num_heads,
            kdim=feat_dim, vdim=feat_dim, batch_first=True)
        self.gru = nn.GRUCell(node_dim, node_dim)
        self.norm_slots = nn.LayerNorm(node_dim)
        self.norm_input = nn.LayerNorm(feat_dim)
        self.proj = nn.Linear(node_dim, node_dim)

    def forward(self, node_queries, point_features, point_mask=None):
        B, M, D = node_queries.shape
        if point_mask is not None:
            any_valid = point_mask.any(dim=-1)
            safe_mask = point_mask.clone()
            if (~any_valid).any():
                safe_mask[~any_valid, 0] = True
            key_padding_mask = ~safe_mask
        else:
            key_padding_mask = None

        pf = self.norm_input(point_features)
        slots = node_queries
        for _ in range(self.num_rounds):
            slots_normed = self.norm_slots(slots)
            updates, _ = self.cross_attn(
                slots_normed, pf, pf, key_padding_mask=key_padding_mask)
            updates_flat = updates.reshape(B * M, D)
            slots_flat = slots.reshape(B * M, D)
            slots = self.gru(updates_flat, slots_flat).reshape(B, M, D)
        return self.proj(slots) + node_queries


# ═══════════════════════════════════════════════════════════
# Block 3. Interaction Context Inference
# ═══════════════════════════════════════════════════════════

class InteractionContextEncoder(nn.Module):
    def __init__(self, node_dim, num_nodes, ctx_dim=128, history_len=5):
        super().__init__()
        self.history_len = history_len
        self.node_dim = node_dim
        self.num_nodes = num_nodes
        self.ctx_dim = ctx_dim
        frame_dim = node_dim * 2 * num_nodes
        self.frame_proj = nn.Sequential(
            nn.Linear(frame_dim, 256), nn.GELU(),
            nn.Linear(256, ctx_dim))
        self.temporal = nn.GRU(
            input_size=ctx_dim, hidden_size=ctx_dim, batch_first=True)
        self.out_norm = nn.LayerNorm(ctx_dim)

    def forward(self, node_history):
        assert len(node_history) >= 1
        B = node_history[0].shape[0]
        device = node_history[0].device
        K = self.history_len
        padded = list(node_history[-K:])
        zero = torch.zeros(B, self.num_nodes, self.node_dim, device=device)
        while len(padded) < K:
            padded.insert(0, zero)
        frames = []
        for i in range(K):
            nodes = padded[i]
            vel = (nodes - padded[i - 1]) if i > 0 else torch.zeros_like(nodes)
            feat = torch.cat([nodes, vel], dim=-1).reshape(B, -1)
            frames.append(self.frame_proj(feat))
        seq = torch.stack(frames, dim=1)
        _, h_n = self.temporal(seq)
        return self.out_norm(h_n.squeeze(0))


# ═══════════════════════════════════════════════════════════
# Block 4. Context-Conditioned Transition — Node-Specific FiLM
# ═══════════════════════════════════════════════════════════

class ContextConditionedTransition(nn.Module):
    def __init__(self, node_dim, num_nodes, ctx_dim=128, num_heads=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.self_attn = nn.MultiheadAttention(
            embed_dim=node_dim, num_heads=num_heads, batch_first=True)
        self.sa_norm = nn.LayerNorm(node_dim)
        self.pre_sa_norm = nn.LayerNorm(node_dim)
        self.ctx_to_gamma_raw = nn.Linear(ctx_dim, num_nodes * node_dim)
        self.ctx_to_beta = nn.Linear(ctx_dim, num_nodes * node_dim)
        self.film_scale = 0.1
        self.gru = nn.GRUCell(node_dim, node_dim)
        self.prior_mu = nn.Linear(node_dim, node_dim)
        self.prior_logvar = nn.Linear(node_dim, node_dim)

    def forward(self, prev_nodes, context):
        B, M, D = prev_nodes.shape
        normed = self.pre_sa_norm(prev_nodes)
        sa_out, _ = self.self_attn(normed, normed, normed)
        nodes = self.sa_norm(prev_nodes + sa_out)
        gamma_raw = self.ctx_to_gamma_raw(context).view(B, M, D)
        beta_raw = self.ctx_to_beta(context).view(B, M, D)
        gamma = 1.0 + self.film_scale * torch.tanh(gamma_raw)
        nodes = gamma * nodes + beta_raw
        prior_nodes = self.gru(
            nodes.reshape(B * M, D),
            prev_nodes.reshape(B * M, D)).reshape(B, M, D)
        mu = self.prior_mu(prior_nodes)
        logvar = self.prior_logvar(prior_nodes).clamp(-6, 2)
        return prior_nodes, mu, logvar


# ═══════════════════════════════════════════════════════════
# Block 5. Evidence + Confidence
# ═══════════════════════════════════════════════════════════

class EvidenceExtractor(nn.Module):
    def __init__(self, node_dim, feat_dim, num_heads=4,
                 compete_alpha=0.2, num_nodes=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        assert node_dim % num_heads == 0
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(feat_dim, node_dim)
        self.v_proj = nn.Linear(feat_dim, node_dim)
        self.out_proj = nn.Linear(node_dim, node_dim)
        self.scale = self.head_dim ** -0.5
        self.compete_alpha = compete_alpha
        self.node_query_bias = nn.Parameter(
            torch.randn(1, num_nodes, node_dim) * 0.02)
        self.confidence_head = nn.Sequential(
            nn.Linear(num_heads * 2 + 1, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, prior_nodes, point_features, point_mask=None):
        B, M, _ = prior_nodes.shape
        N = point_features.shape[1]
        H, d = self.num_heads, self.head_dim

        Q = self.q_proj(prior_nodes + self.node_query_bias[:, :M, :]).view(B, M, H, d).transpose(1, 2)
        K = self.k_proj(point_features).view(B, N, H, d).transpose(1, 2)
        V = self.v_proj(point_features).view(B, N, H, d).transpose(1, 2)

        logits = (Q @ K.transpose(-2, -1)) * self.scale
        logits = logits.clamp(-30, 30)

        if point_mask is not None:
            vm = point_mask[:, None, None, :].expand_as(logits)
            any_valid = point_mask.any(dim=-1)
            has_invalid = (~any_valid).any()
            logits_masked = logits.masked_fill(~vm, float("-inf"))
            logit_max_raw = logits_masked.max(dim=-1).values
            logits_zeroed = logits * vm.float()
            valid_count = vm.float().sum(dim=-1).clamp(min=1)
            logit_mean_raw = logits_zeroed.sum(dim=-1) / valid_count
            n_valid_ratio = point_mask.float().sum(dim=-1) / N
            n_valid = n_valid_ratio.view(B, 1, 1).expand(B, 1, M)
            if has_invalid:
                inv = (~any_valid)[:, None, None].expand_as(logit_max_raw)
                logit_max_raw = logit_max_raw.masked_fill(inv, 0.0)
                logit_mean_raw = logit_mean_raw.masked_fill(inv, 0.0)
            logit_max = logit_max_raw
            logit_mean = logit_mean_raw
            logits = logits.masked_fill(~vm, float("-inf"))
        else:
            any_valid = torch.ones(B, dtype=torch.bool, device=prior_nodes.device)
            has_invalid = False
            logit_max = logits.max(dim=-1).values
            logit_mean = logits.mean(dim=-1)
            n_valid = torch.ones(B, 1, M, device=prior_nodes.device)

        conf_input = torch.cat([
            logit_max.permute(0, 2, 1),
            logit_mean.permute(0, 2, 1),
            n_valid.permute(0, 2, 1),
        ], dim=-1)
        confidence = self.confidence_head(conf_input)

        attn_np = torch.nan_to_num(logits.softmax(dim=-1), nan=0.0)
        if self.compete_alpha > 0:
            attn_pn = torch.nan_to_num(logits.softmax(dim=-2), nan=0.0)
            attn = (1 - self.compete_alpha) * attn_np + self.compete_alpha * attn_pn
        else:
            attn = attn_np

        out = (attn @ V).transpose(1, 2).reshape(B, M, -1)
        evidence = self.out_proj(out)

        if has_invalid:
            mask_f = (~any_valid)[:, None, None].float()
            confidence = confidence * (1 - mask_f)
            evidence = evidence * (1 - mask_f)
        return evidence, confidence


class PosteriorUpdate(nn.Module):
    def __init__(self, node_dim, num_heads=4):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim), nn.GELU(),
            nn.Linear(node_dim, node_dim))
        self.fuse_norm = nn.LayerNorm(node_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=node_dim, num_heads=num_heads, batch_first=True)
        self.sa_norm = nn.LayerNorm(node_dim)
        self.post_mu = nn.Linear(node_dim, node_dim)
        self.post_logvar = nn.Linear(node_dim, node_dim)

    def forward(self, prior, evidence, confidence, training=True):
        fused = self.fuse(torch.cat([prior, evidence], dim=-1))
        gated = confidence * fused + (1 - confidence) * prior
        gated = self.fuse_norm(gated)
        sa_out, _ = self.self_attn(gated, gated, gated)
        coordinated = self.sa_norm(gated + sa_out)
        mu = self.post_mu(coordinated)
        logvar = self.post_logvar(coordinated).clamp(-6, 2)
        posterior = reparameterize(mu, logvar, training=training)
        return posterior, mu, logvar


# ═══════════════════════════════════════════════════════════
# Decoders
# ═══════════════════════════════════════════════════════════

class NodeWiseDecoder(nn.Module):
    def __init__(self, node_dim, num_nodes, num_points=128, point_dim=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_points = num_points
        self.point_dim = point_dim
        self.points_per_node = (num_points + num_nodes - 1) // num_nodes

        self.decoder = nn.Sequential(
            nn.Linear(node_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, self.points_per_node * point_dim))

    def forward(self, nodes):
        B, M, D = nodes.shape
        per_node = self.decoder(nodes)
        per_node = per_node.reshape(B, M * self.points_per_node, self.point_dim)
        return per_node[:, :self.num_points, :]


# ═══════════════════════════════════════════════════════════
# Node Temporal Readout + Radar Semantic Head
# (g_radar 생성용 — 여전히 필요)
# ═══════════════════════════════════════════════════════════

class NodeTemporalReadout(nn.Module):
    """(B, T, M, D) → (B, M, D)"""
    def __init__(self, node_dim, num_heads=4):
        super().__init__()
        self.temporal_pool = nn.MultiheadAttention(
            embed_dim=node_dim, num_heads=num_heads, batch_first=True)
        self.temporal_query = nn.Parameter(torch.randn(1, 1, node_dim) * 0.02)
        self.out_norm = nn.LayerNorm(node_dim)

    def forward(self, node_seq, temporal_mask=None):
        B, T, M, D = node_seq.shape
        x = node_seq.permute(0, 2, 1, 3).contiguous().view(B * M, T, D)
        q = self.temporal_query.expand(B * M, -1, -1)
        if temporal_mask is not None:
            kp = ~temporal_mask[:, None, :].expand(B, M, T).reshape(B * M, T)
        else:
            kp = None
        node_emb, _ = self.temporal_pool(q, x, x, key_padding_mask=kp)
        return self.out_norm(node_emb.squeeze(1)).view(B, M, D)


class RadarSemanticHead(nn.Module):
    """(B, M, D) → (B, K, out_dim) semantic tokens + (B, out_dim) global"""
    def __init__(self, node_dim, out_dim=512, num_queries=4,
                 num_heads=4, num_rounds=3, compete_alpha=0.3):
        super().__init__()
        self.num_queries = num_queries
        self.num_rounds = num_rounds
        self.compete_alpha = compete_alpha

        self.token_queries = nn.Parameter(
            self._init_orthogonal(num_queries, node_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=node_dim, num_heads=num_heads, batch_first=True)
        self.gru = nn.GRUCell(node_dim, node_dim)
        self.norm_q = nn.LayerNorm(node_dim)
        self.norm_k = nn.LayerNorm(node_dim)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(node_dim), nn.Linear(node_dim, out_dim))
        self.global_query = nn.Parameter(torch.randn(1, 1, node_dim) * 0.02)
        self.global_pool = nn.MultiheadAttention(
            embed_dim=node_dim, num_heads=num_heads, batch_first=True)
        self.global_proj = nn.Sequential(
            nn.LayerNorm(node_dim), nn.Linear(node_dim, out_dim))

    @staticmethod
    def _init_orthogonal(num_queries, dim):
        if num_queries <= dim:
            rand = torch.randn(dim, num_queries)
            q, _ = torch.linalg.qr(rand)
            return q[:, :num_queries].t().unsqueeze(0)
        return torch.randn(1, num_queries, dim) * 0.5

    def forward(self, node_emb):
        B, M, D = node_emb.shape
        K = self.num_queries
        kv = self.norm_k(node_emb)
        slots = self.token_queries.expand(B, -1, -1)

        for _ in range(self.num_rounds):
            slots_normed = self.norm_q(slots)
            updates, _ = self.cross_attn(
                slots_normed, kv, kv, need_weights=True,
                average_attn_weights=True)
            slots = self.gru(
                updates.reshape(B * K, D),
                slots.reshape(B * K, D)
            ).reshape(B, K, D)

        q = self.norm_q(slots)
        scores = torch.bmm(q, kv.transpose(1, 2)) / (D ** 0.5)
        attn_qn = scores.softmax(dim=-1)
        attn_nq = scores.softmax(dim=-2)
        attn = (1 - self.compete_alpha) * attn_qn + self.compete_alpha * attn_nq
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        sem_tokens = torch.bmm(attn, kv)
        sem_tokens = self.token_proj(sem_tokens)

        q_g = self.global_query.expand(B, -1, -1)
        g, _ = self.global_pool(q_g, node_emb, node_emb)
        g = self.global_proj(g.squeeze(1))

        return sem_tokens, g, attn


class NodeAwareReadout(nn.Module):
    """(B, T, M, D) → semantic tokens (B, K, out_dim) + global (B, out_dim)"""
    def __init__(self, node_dim, out_dim=512,
                 num_semantic_queries=4, num_heads=4):
        super().__init__()
        self.node_temporal = NodeTemporalReadout(node_dim, num_heads)
        self.radar_semantic = RadarSemanticHead(
            node_dim, out_dim, num_semantic_queries, num_heads)

    def forward(self, node_seq, temporal_mask=None):
        node_emb = self.node_temporal(node_seq, temporal_mask)
        sem_tokens, g_radar, attn = self.radar_semantic(node_emb)
        return {
            "node_emb": node_emb,
            "radar_tokens": sem_tokens,
            "g_radar": g_radar,
            "token_attn": attn,
        }


# ═══════════════════════════════════════════════════════════
# Phase Head + Pseudo Labeler (unchanged)
# ═══════════════════════════════════════════════════════════

class PhaseHead(nn.Module):
    NUM_PHASES = 4
    def __init__(self, node_dim, num_nodes):
        super().__init__()
        self.frame_enc = nn.Sequential(
            nn.Linear(node_dim * num_nodes, 256), nn.GELU(),
            nn.Linear(256, 128))
        self.temporal = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=5, padding=2), nn.GELU())
        self.classifier = nn.Linear(64, self.NUM_PHASES)

    def forward(self, node_seq):
        B, T, M, D = node_seq.shape
        frame = self.frame_enc(node_seq.reshape(B, T, -1))
        ctx = self.temporal(frame.transpose(1, 2)).transpose(1, 2)
        return self.classifier(ctx)


class InteractionPseudoLabeler:
    NO_INTERACTION, APPROACH, CONTACT, MANIPULATION = 0, 1, 2, 3
    HAND_JOINTS = [20, 21]

    def __init__(self, vel_thr=0.15, contact_thr=0.02,
                 accel_thr=0.1, min_frames=3):
        self.vel_thr = vel_thr
        self.contact_thr = contact_thr
        self.accel_thr = accel_thr
        self.min_frames = min_frames

    @torch.no_grad()
    def generate(self, joints):
        if isinstance(joints, torch.Tensor):
            joints = joints.detach().cpu()
        T = joints.shape[0]
        hand = joints[:, self.HAND_JOINTS].mean(dim=1)
        vel = torch.zeros(T)
        if T > 1:
            vel[1:] = (hand[1:] - hand[:-1]).norm(dim=-1)
            vel[0] = vel[1]
        accel = torch.zeros(T)
        if T > 2:
            accel[1:-1] = (vel[2:] - vel[:-2]).abs() / 2
            accel[0], accel[-1] = accel[1], accel[-2]
        torso = joints[:, 0]
        reach = (hand - torso).norm(dim=-1)
        labels = torch.zeros(T, dtype=torch.long)
        conf = torch.full((T,), 0.5)
        for t in range(T):
            if accel[t] > self.accel_thr and vel[t] < self.contact_thr:
                labels[t], conf[t] = self.CONTACT, min(0.9, 0.5 + accel[t]*3)
            elif vel[t] > self.vel_thr and reach[t] > reach.median():
                labels[t], conf[t] = self.APPROACH, min(0.8, 0.4 + vel[t]*2)
            elif vel[t] < self.contact_thr * 2:
                if t > 0 and labels[t-1] in (self.CONTACT, self.MANIPULATION):
                    labels[t], conf[t] = self.MANIPULATION, 0.6
                else:
                    labels[t], conf[t] = self.NO_INTERACTION, 0.7
            else:
                labels[t], conf[t] = self.NO_INTERACTION, 0.6
        return self._smooth(labels), conf

    def _smooth(self, labels):
        T, mf = len(labels), self.min_frames
        if T < mf * 2:
            return labels
        out = labels.clone()
        i = 0
        while i < T:
            j = i
            while j < T and labels[j] == labels[i]:
                j += 1
            if j - i < mf:
                out[i:j] = labels[max(0, i-1)]
            i = j
        return out


# ═══════════════════════════════════════════════════════════
# Node Diversity Loss (diagnostic / optional regularizer)
# ═══════════════════════════════════════════════════════════

def node_diversity_loss(node_seq, temporal_mask=None):
    B, T, M, D = node_seq.shape
    node_norm = F.normalize(node_seq, dim=-1)
    sim = torch.bmm(
        node_norm.reshape(B * T, M, D),
        node_norm.reshape(B * T, M, D).transpose(1, 2)
    ).reshape(B, T, M, M)
    eye = torch.eye(M, device=sim.device).view(1, 1, M, M)
    off_diag = (sim * (1 - eye)).pow(2)
    if temporal_mask is not None:
        mask = temporal_mask.float().view(B, T, 1, 1)
        loss = (off_diag * mask).sum() / (mask.sum() * M * (M - 1)).clamp(1)
    else:
        loss = off_diag.sum() / (B * T * M * (M - 1))
    return loss


# ═══════════════════════════════════════════════════════════
# Full Model — v8.0 (HY-Motion 201D Direct Prediction)
# ═══════════════════════════════════════════════════════════

class LatentGraphDynamicsModel(nn.Module):
    """
    v8.0: HY-Motion 201D Direct Prediction.

    변경 핵심:
      - contrastive 제거 (MotionSemanticHead, projection heads, queue 모두 제거)
      - hymotion_head 추가: g_radar(512D) → 201D 직접 예측
      - loss: L_obs + β·L_KL + λ_latent·MSE(pred_201, gt_201)
      - 추론 시: pred_201 → HY-Motion decode_motion_from_latent() → full motion

    기존 유지:
      - LatentGraph 전체 (노드, context, transition, evidence, posterior)
      - NodeAwareReadout (g_radar 생성)
      - NodeWiseDecoder (L_obs)
      - PhaseHead (optional)
    """

    def __init__(self, config: dict):
        super().__init__()
        cfg = config.get("latent_graph", config)

        self.num_nodes = cfg.get("num_nodes", 8)
        self.node_dim = cfg.get("node_dim", 256)
        self.feat_dim = cfg.get("point_feat_dim", 256)
        self.point_in_dim = cfg.get("point_in_dim", 4)
        self.num_points = cfg.get("num_points", 128)
        self.out_dim = cfg.get("out_dim", 512)
        self.ctx_dim = cfg.get("ctx_dim", 128)
        self.ctx_history_len = cfg.get("ctx_history_len", 5)
        num_heads = cfg.get("num_heads", 4)
        self.num_semantic_queries = cfg.get("num_semantic_queries", 4)

        # ── Loss weights ──
        self.beta_kl = cfg.get("beta_kl", 0.1)
        self.lambda_latent = cfg.get("lambda_latent", 1.0)  # ★ 201D MSE weight
        self.lambda_phase = cfg.get("lambda_phase", 0.0)
        self.lambda_diversity = cfg.get("lambda_diversity", 0.0)

        # ── Orthogonal node query init ──
        self.node_queries = nn.Parameter(self._init_orthogonal_queries(
            self.num_nodes, self.node_dim))

        # ── Core components (unchanged) ──
        self.obs_encoder = ObservationEncoder(
            in_dim=self.point_in_dim,
            hidden_dims=cfg.get("encoder_hidden", [64, 128]),
            out_dim=self.feat_dim)

        self.node_init = LatentNodeInitializer(
            self.node_dim, self.feat_dim, num_heads,
            num_rounds=cfg.get("init_rounds", 3))

        self.context_encoder = InteractionContextEncoder(
            self.node_dim, self.num_nodes,
            ctx_dim=self.ctx_dim, history_len=self.ctx_history_len)

        self.transition = ContextConditionedTransition(
            self.node_dim, self.num_nodes, self.ctx_dim, num_heads)

        self.evidence_extractor = EvidenceExtractor(
            node_dim=self.node_dim, feat_dim=self.feat_dim,
            num_heads=num_heads,
            compete_alpha=cfg.get("compete_alpha", 0.2),
            num_nodes=self.num_nodes)

        self.posterior_update = PosteriorUpdate(self.node_dim, num_heads)

        self.decoder = NodeWiseDecoder(
            self.node_dim, self.num_nodes, self.num_points, 3)

        # ── Readout (unchanged — produces g_radar) ──
        self.readout = NodeAwareReadout(
            node_dim=self.node_dim,
            out_dim=self.out_dim,
            num_semantic_queries=self.num_semantic_queries,
            num_heads=num_heads)

        # ═══════════════════════════════════════════════════
        # ★ v8.0 NEW: HY-Motion 201D prediction head
        # ═══════════════════════════════════════════════════
        self.hymotion_latent_dim = cfg.get("hymotion_latent_dim", 201)
        self.hymotion_head = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.hymotion_latent_dim),
        )

        # ═══════════════════════════════════════════════════
        # REMOVED (v7.0 → v8.0):
        #   - MotionSemanticHead
        #   - radar_proj, motion_proj
        #   - motion_queue, queue_ptr
        # ═══════════════════════════════════════════════════

        if self.lambda_phase > 0:
            self.phase_head = PhaseHead(self.node_dim, self.num_nodes)
        else:
            self.phase_head = None

        self.diagnostics = NodeDiagnostics()

    @staticmethod
    def _init_orthogonal_queries(num_nodes, node_dim):
        if num_nodes <= node_dim:
            rand = torch.randn(node_dim, num_nodes)
            q, _ = torch.linalg.qr(rand)
            queries = q[:, :num_nodes].t().unsqueeze(0)
        else:
            queries = torch.randn(1, num_nodes, node_dim) * 0.5
        return queries

    def _point_mask(self, Y):
        return Y[..., :3].norm(dim=-1) > 1e-6

    def forward_sequence(self, Y, temporal_mask=None):
        """Core forward — unchanged from v7.0."""
        B, T, N, _ = Y.shape
        device = Y.device
        is_train = self.training

        if temporal_mask is not None:
            if not temporal_mask[:, 0].all():
                temporal_mask[:, 0] = True
            for b in range(B):
                valid_len = temporal_mask[b].long().sum()
                temporal_mask[b, valid_len:] = False

        U = self.obs_encoder(Y)
        pmask = self._point_mask(Y)

        if temporal_mask is not None:
            frame_has_points = pmask.any(dim=-1)
            temporal_mask = temporal_mask & frame_has_points
            for b in range(B):
                valid_len = temporal_mask[b].long().sum()
                temporal_mask[b, valid_len:] = False
            if not temporal_mask[:, 0].all():
                temporal_mask[:, 0] = True

        queries = self.node_queries.expand(B, -1, -1)
        nodes = self.node_init(queries, U[:, 0], pmask[:, 0])

        node_hist = []
        prev_conf = torch.zeros(B, self.num_nodes, 1, device=device)
        H_node, H_recon, H_conf = [], [], []
        H_context, H_post_lv = [], []
        kl_sum = torch.tensor(0.0, device=device)
        kl_weight = torch.tensor(0.0, device=device)

        for t in range(T):
            U_t = U[:, t]
            m_t = pmask[:, t]
            if temporal_mask is not None:
                valid_t = temporal_mask[:, t]
            else:
                valid_t = torch.ones(B, dtype=torch.bool, device=device)
            prev_nodes = nodes

            if t == 0:
                ev, conf = self.evidence_extractor(nodes, U_t, m_t)
                new_nodes, post_mu, post_lv = self.posterior_update(
                    nodes, ev, conf, training=is_train)
                c_t = torch.zeros(B, self.ctx_dim, device=device)
            else:
                c_t = self.context_encoder(node_hist[-self.ctx_history_len:])
                prior_nodes, prior_mu, prior_lv = self.transition(nodes, c_t)
                ev, conf = self.evidence_extractor(prior_nodes, U_t, m_t)
                new_nodes, post_mu, post_lv = self.posterior_update(
                    prior_nodes, ev, conf, training=is_train)
                kl_per_sample = kl_divergence_per_sample(
                    post_mu, post_lv, prior_mu, prior_lv)
                kl_sum = kl_sum + (kl_per_sample * valid_t.float()).sum()
                kl_weight = kl_weight + valid_t.float().sum()

            valid_f = valid_t[:, None, None].float()
            nodes = valid_f * new_nodes + (1 - valid_f) * prev_nodes
            conf = valid_f * conf + (1 - valid_f) * prev_conf
            prev_conf = conf.detach()

            recon = self.decoder(nodes)
            node_hist.append(nodes)
            if len(node_hist) > self.ctx_history_len:
                node_hist.pop(0)
            H_node.append(nodes)
            H_recon.append(recon)
            H_conf.append(conf)
            H_context.append(c_t)
            H_post_lv.append(post_lv)

        node_seq = torch.stack(H_node, 1)
        recon_seq = torch.stack(H_recon, 1)
        conf_seq = torch.stack(H_conf, 1)
        ctx_seq = torch.stack(H_context, 1)
        post_lv_seq = torch.stack(H_post_lv, 1)

        rd = self.readout(node_seq, temporal_mask)

        # ★ v8.0: 201D prediction from g_radar
        pred_hymotion_latent = self.hymotion_head(rd["g_radar"])  # (B, 201)

        phase_logits = None
        if self.phase_head is not None:
            phase_logits = self.phase_head(node_seq)

        kl = kl_sum / kl_weight.clamp(min=1.0)

        return {
            "node_history": node_seq,
            "recon_sequence": recon_seq,
            "confidence": conf_seq,
            "context_history": ctx_seq,
            "post_logvar": post_lv_seq,
            "g_radar": rd["g_radar"],
            "radar_tokens": rd["radar_tokens"],
            "node_emb": rd["node_emb"],
            "token_attn": rd["token_attn"],
            "pred_hymotion_latent": pred_hymotion_latent,  # ★ NEW
            "phase_logits": phase_logits,
            "kl": kl,
        }

    def forward(self, point_cloud,
                motion_latent_201=None,
                temporal_mask=None,
                phase_labels=None,
                phase_confidence=None):
        """
        v8.0 forward — simplified.

        Args:
            point_cloud:       (B, T, N, D_in) radar point cloud
            motion_latent_201: (B, T_m, 201) HY-Motion latent (frozen GT)
                               또는 None (inference)
            temporal_mask:     (B, T) bool — radar temporal mask
            phase_labels:      (B, T) long — optional pseudo labels
            phase_confidence:  (B, T) float — optional pseudo confidence
        """
        out = self.forward_sequence(point_cloud, temporal_mask)
        device = point_cloud.device

        # ── L_obs: point cloud reconstruction ──
        loss_obs = self._centered_chamfer(
            out["recon_sequence"], point_cloud[..., :3], temporal_mask)

        # ── L_KL ──
        loss_kl = out["kl"]

        # ── L_div (optional diagnostic) ──
        if self.lambda_diversity > 0:
            metric_div = node_diversity_loss(out["node_history"], temporal_mask)
        else:
            with torch.no_grad():
                metric_div = node_diversity_loss(out["node_history"], temporal_mask)

        # ═══════════════════════════════════════════════════
        # ★ v8.0: L_latent — 201D direct MSE
        # ═══════════════════════════════════════════════════
        loss_latent = torch.tensor(0.0, device=device)
        if motion_latent_201 is not None:
            # motion_latent_201: (B, T_m, 201) → 시간 평균하여 (B, 201)
            target_201 = motion_latent_201.mean(dim=1)  # (B, 201)
            pred_201 = out["pred_hymotion_latent"]       # (B, 201)
            loss_latent = F.mse_loss(pred_201, target_201)

        # ── L_phase (optional) ──
        loss_phase = torch.tensor(0.0, device=device)
        if phase_labels is not None and out["phase_logits"] is not None:
            pl = out["phase_logits"]
            Tm = min(pl.shape[1], phase_labels.shape[1])
            lf = pl[:, :Tm].reshape(-1, pl.shape[-1])
            lb = phase_labels[:, :Tm].reshape(-1).to(device)
            if temporal_mask is not None:
                pm = temporal_mask[:, :Tm].reshape(-1)
                lf = lf[pm]
                lb = lb[pm]
            else:
                pm = None
            if lf.shape[0] > 0:
                if phase_confidence is not None:
                    w = phase_confidence[:, :Tm].reshape(-1).to(device)
                    if pm is not None:
                        w = w[pm]
                    loss_phase = (F.cross_entropy(
                        lf, lb, reduction="none") * w).mean()
                else:
                    loss_phase = F.cross_entropy(lf, lb)

        # ── Total loss ──
        loss = (loss_obs
                + self.beta_kl * loss_kl
                + self.lambda_latent * loss_latent
                + self.lambda_phase * loss_phase
                + self.lambda_diversity * metric_div)

        return {
            "loss": loss,
            "loss_obs": loss_obs,
            "loss_kl": loss_kl,
            "loss_latent": loss_latent,         # ★ NEW (replaces loss_rm_*)
            "loss_phase": loss_phase,
            "metric_div": metric_div,
            "g_radar": out["g_radar"],
            "pred_hymotion_latent": out["pred_hymotion_latent"],  # ★ NEW
            "radar_tokens": out["radar_tokens"],
            "node_emb": out["node_emb"],
            "token_attn": out["token_attn"],
            "confidence": out["confidence"],
            "node_history": out["node_history"],
            "phase_logits": out["phase_logits"],
        }

    def _centered_chamfer(self, pred, gt, temporal_mask=None):
        B, T = pred.shape[:2]
        gt_valid = gt.norm(dim=-1) > 1e-6
        total, count = 0.0, 0
        for t in range(T):
            fv = temporal_mask[:, t] if temporal_mask is not None else \
                torch.ones(B, device=pred.device, dtype=torch.bool)
            for b in range(B):
                if not fv[b]:
                    continue
                vm = gt_valid[b, t]
                if vm.sum() < 2:
                    continue
                g, p = gt[b, t, vm], pred[b, t]
                g_n = g - g.mean(0, keepdim=True)
                p_n = p - p.mean(0, keepdim=True)
                d = (p_n.unsqueeze(1) - g_n.unsqueeze(0)).pow(2).sum(-1)
                total += d.min(1).values.mean() + d.min(0).values.mean()
                count += 1
        return total / max(count, 1)

    # ═══════════════════════════════════════════════════
    # Inference methods
    # ═══════════════════════════════════════════════════

    @torch.no_grad()
    def encode(self, point_cloud, temporal_mask=None):
        """Global radar embedding for evaluation."""
        return self.forward_sequence(point_cloud, temporal_mask)["g_radar"]

    @torch.no_grad()
    def predict_hymotion_latent(self, point_cloud, temporal_mask=None):
        """
        ★ v8.0 핵심 추론: radar → 201D HY-Motion latent 예측.

        Returns:
            pred_201: (B, 201) — HY-Motion decode_motion_from_latent()에 입력 가능
        """
        out = self.forward_sequence(point_cloud, temporal_mask)
        return out["pred_hymotion_latent"]

    @torch.no_grad()
    def encode_tokens(self, point_cloud, temporal_mask=None):
        """Semantic tokens for token-level evaluation."""
        out = self.forward_sequence(point_cloud, temporal_mask)
        return out["radar_tokens"], out["g_radar"]

    def get_diagnostics(self, model_output, temporal_mask=None):
        return self.diagnostics.compute(model_output, temporal_mask)
