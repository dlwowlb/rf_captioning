"""
Latent Graph Dynamics — v7.0 (Node-Aware Semantic Readout)

v6.0 → v7.0 changes:
  [new-1] NodeTemporalReadout: 각 node의 trajectory를 따로 temporal pool
  [new-2] RadarSemanticHead: K개 semantic query가 node subset을 읽음
  [new-3] MotionSemanticHead: motion sequence에서 K개 semantic token 추출
  [new-4] NodeAwareReadout: new-1 + new-2 통합
  [new-5] Token-level contrastive: semantic token끼리 alignment
  [new-6] Global + token dual contrastive loss

기존 유지:
  [fix-1] Node-specific FiLM in transition
  [fix-2] Orthogonal node query init
  [fix-3] Iterative slot competition init
  [fix-A] Node query bias in EvidenceExtractor
  [fix-6] Momentum queue for contrastive
  [fix-7] NodeDiagnostics

핵심 설계:
  기존: (B,T,M,D) → g_radar (B,512) 하나 → g_motion 하나와 contrastive
  변경: (B,T,M,D) → node별 temporal pool (B,M,D)
        → K개 semantic token (B,K,512) + global (B,512)
        → motion도 K개 token + global
        → global contrastive + token-level contrastive

  이 구조에서 node가 서로 다른 역할을 해야만
  semantic token이 다양해지고 token-level loss가 줄어듦.
  → diversity loss 없이도 자연스러운 분화 유도.

Loss: L_obs + β·L_KL + λ_m·L_rm_global + λ_t·L_rm_token [+ λ_ph·L_phase] [+ λ_d·L_div]
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
# Contrastive Loss Functions
# ═══════════════════════════════════════════════════════════

def global_contrastive_loss(g_r, g_m, temp=0.3):
    """Symmetric InfoNCE on global vectors."""
    g_r = F.normalize(g_r, dim=-1)
    g_m = F.normalize(g_m, dim=-1)
    sim = g_r @ g_m.t() / temp
    lab = torch.arange(g_r.size(0), device=g_r.device)
    return 0.5 * (F.cross_entropy(sim, lab) + F.cross_entropy(sim.t(), lab))


def token_contrastive_loss(r_tokens, m_tokens, temp=0.3):
    """
    Token-level contrastive: query index별로 positive pair.
    r_tokens: (B, K, D), m_tokens: (B, K, D)
    같은 query index끼리 같은 sample이면 positive.
    """
    B, K, D = r_tokens.shape
    r = F.normalize(r_tokens, dim=-1).permute(1, 0, 2)  # (K, B, D)
    m = F.normalize(m_tokens, dim=-1).permute(1, 0, 2)  # (K, B, D)

    losses = []
    lab = torch.arange(B, device=r.device)
    for k in range(K):
        sim = r[k] @ m[k].t() / temp  # (B, B)
        l = 0.5 * (F.cross_entropy(sim, lab) + F.cross_entropy(sim.t(), lab))
        losses.append(l)
    return torch.stack(losses).mean()


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
# Block 5. Evidence + Confidence — Node Query Bias + Dual Softmax
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
        # [fix-A] per-node query bias
        self.node_query_bias = nn.Parameter(
            torch.randn(1, num_nodes, node_dim) * 0.02)
        self.confidence_head = nn.Sequential(
            nn.Linear(num_heads * 2 + 1, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, prior_nodes, point_features, point_mask=None):
        B, M, _ = prior_nodes.shape
        N = point_features.shape[1]
        H, d = self.num_heads, self.head_dim

        # [fix-A] bias 적용
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
# Decoder
# ═══════════════════════════════════════════════════════════

class PointCloudDecoder(nn.Module):
    def __init__(self, node_dim, num_nodes, num_points=128, point_dim=3):
        super().__init__()
        self.num_points = num_points
        self.point_dim = point_dim
        self.decoder = nn.Sequential(
            nn.Linear(node_dim * num_nodes, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, num_points * point_dim))

    def forward(self, nodes):
        B = nodes.shape[0]
        return self.decoder(nodes.reshape(B, -1)).reshape(
            B, self.num_points, self.point_dim)

class NodeWiseDecoder(nn.Module):
    """
    기존: nodes.reshape(B, -1) → shared MLP → 128 points
         → 노드가 다를 필요 없음

    변경: 각 노드가 독립적으로 자기 포인트를 생성
         → 노드가 다른 영역을 맡아야만 L_obs가 줄어듦
    
    Shared MLP를 node별로 적용: 같은 weight지만 입력(node embedding)이 다르면
    출력(points)도 다름. 노드가 같으면 같은 포인트를 중복 생성 → Chamfer 증가.
    """
    def __init__(self, node_dim, num_nodes, num_points=128, point_dim=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_points = num_points
        self.point_dim = point_dim
        self.points_per_node = (num_points + num_nodes - 1) // num_nodes  # ceil

        self.decoder = nn.Sequential(
            nn.Linear(node_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, self.points_per_node * point_dim))

    def forward(self, nodes):
        # nodes: (B, M, D)
        B, M, D = nodes.shape
        per_node = self.decoder(nodes)                          # (B, M, ppn*3)
        per_node = per_node.reshape(B, M * self.points_per_node, self.point_dim)
        return per_node[:, :self.num_points, :]                 # (B, N, 3)


# ═══════════════════════════════════════════════════════════
# [new-1] Node Temporal Readout
# ═══════════════════════════════════════════════════════════

class NodeTemporalReadout(nn.Module):
    """
    (B, T, M, D) → (B, M, D)
    각 node의 시간축 trajectory를 따로 읽어서 node별 embedding 생성.
    """
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


# ═══════════════════════════════════════════════════════════
# [new-2] Radar Semantic Head
# ═══════════════════════════════════════════════════════════

class RadarSemanticHead(nn.Module):
    """
    (B, M, D) → (B, K, out_dim) semantic tokens + (B, out_dim) global
    
    변경:
      - orthogonal query init (query끼리 처음부터 다른 방향)
      - iterative slot competition (query가 서로 다른 노드를 읽도록 경쟁)
      - softmax(dim=-2) 블렌딩 (각 노드가 하나의 query에 "선택"됨)
    """
    def __init__(self, node_dim, out_dim=512, num_queries=4,
                 num_heads=4, num_rounds=3, compete_alpha=0.3):
        super().__init__()
        self.num_queries = num_queries
        self.num_rounds = num_rounds
        self.compete_alpha = compete_alpha

        # ★ Orthogonal query init
        self.token_queries = nn.Parameter(
            self._init_orthogonal(num_queries, node_dim))

        # Iterative refinement (slot attention style)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=node_dim, num_heads=num_heads, batch_first=True)
        self.gru = nn.GRUCell(node_dim, node_dim)
        self.norm_q = nn.LayerNorm(node_dim)
        self.norm_k = nn.LayerNorm(node_dim)

        self.token_proj = nn.Sequential(
            nn.LayerNorm(node_dim), nn.Linear(node_dim, out_dim))

        # Global (단일 query — 변경 없음)
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
            return q[:, :num_queries].t().unsqueeze(0)  # (1, K, D)
        return torch.randn(1, num_queries, dim) * 0.5

    def forward(self, node_emb):
        B, M, D = node_emb.shape
        K = self.num_queries

        kv = self.norm_k(node_emb)
        slots = self.token_queries.expand(B, -1, -1)

        # ★ Iterative slot competition
        for _ in range(self.num_rounds):
            slots_normed = self.norm_q(slots)

            # Raw attention scores
            updates, raw_attn = self.cross_attn(
                slots_normed, kv, kv, need_weights=True,
                average_attn_weights=True)
            # raw_attn: (B, K, M)

            # GRU update
            slots = self.gru(
                updates.reshape(B * K, D),
                slots.reshape(B * K, D)
            ).reshape(B, K, D)

        # ★ 최종 attention을 dual softmax로 재계산
        # query-node score 직접 계산
        q = self.norm_q(slots)
        scores = torch.bmm(q, kv.transpose(1, 2))  # (B, K, M)
        scores = scores / (D ** 0.5)

        attn_qn = scores.softmax(dim=-1)            # query → node (기존)
        attn_nq = scores.softmax(dim=-2)             # node → query (경쟁)
        attn = (1 - self.compete_alpha) * attn_qn + self.compete_alpha * attn_nq
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        sem_tokens = torch.bmm(attn, kv)            # (B, K, D)
        sem_tokens = self.token_proj(sem_tokens)     # (B, K, out_dim)

        # Global
        q_g = self.global_query.expand(B, -1, -1)
        g, _ = self.global_pool(q_g, node_emb, node_emb)
        g = self.global_proj(g.squeeze(1))           # (B, out_dim)

        return sem_tokens, g, attn


# ═══════════════════════════════════════════════════════════
# [new-3] Motion Semantic Head
# ═══════════════════════════════════════════════════════════

class MotionSemanticHead(nn.Module):
    """
    (B, T', D_m) → (B, K, out_dim) tokens + (B, out_dim) global
    """
    def __init__(self, motion_dim, out_dim=512, num_queries=4, num_heads=4):
        super().__init__()
        self.token_queries = nn.Parameter(
            torch.randn(1, num_queries, motion_dim) * 0.02)
        self.token_pool = nn.MultiheadAttention(
            embed_dim=motion_dim, num_heads=num_heads, batch_first=True)
        self.global_query = nn.Parameter(torch.randn(1, 1, motion_dim) * 0.02)
        self.global_pool = nn.MultiheadAttention(
            embed_dim=motion_dim, num_heads=num_heads, batch_first=True)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(motion_dim), nn.Linear(motion_dim, out_dim))
        self.global_proj = nn.Sequential(
            nn.LayerNorm(motion_dim), nn.Linear(motion_dim, out_dim))

    def forward(self, motion_seq, motion_mask=None):
        B = motion_seq.shape[0]
        kp = ~motion_mask if motion_mask is not None else None
        q_tok = self.token_queries.expand(B, -1, -1)
        tok, _ = self.token_pool(q_tok, motion_seq, motion_seq, key_padding_mask=kp)
        q_g = self.global_query.expand(B, -1, -1)
        g, _ = self.global_pool(q_g, motion_seq, motion_seq, key_padding_mask=kp)
        return self.token_proj(tok), self.global_proj(g.squeeze(1))


# ═══════════════════════════════════════════════════════════
# [new-4] Node-Aware Readout (replaces SequenceReadout)
# ═══════════════════════════════════════════════════════════

class NodeAwareReadout(nn.Module):
    """
    (B, T, M, D) → semantic tokens (B, K, out_dim) + global (B, out_dim)

    기존 SequenceReadout: 모든 node를 한번에 평균 → 하나의 벡터
    NodeAwareReadout: node별 temporal pool → K개 semantic token + global
    """
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
# Full Model — v7.0
# ═══════════════════════════════════════════════════════════

class LatentGraphDynamicsModel(nn.Module):
    """
    v7.0: Node-Aware Semantic Readout + Token-Level Contrastive.

    기존 단일 g_radar ↔ g_motion 대신:
      radar: (B,T,M,D) → node temporal pool → K semantic tokens + global
      motion: (B,T',D_m) → K semantic tokens + global
      loss: global contrastive + token-level contrastive

    Loss: L_obs + β·L_KL + λ_m·L_rm_global + λ_t·L_rm_token
          [+ λ_ph·L_phase] [+ λ_d·L_div]
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

        self.beta_kl = cfg.get("beta_kl", 0.1)
        self.lambda_motion = cfg.get("lambda_motion", 1.0)
        self.lambda_token = cfg.get("lambda_token", 0.5)
        self.lambda_phase = cfg.get("lambda_phase", 0.0)
        self.lambda_diversity = cfg.get("lambda_diversity", 0.0)
        self.contrastive_temp = cfg.get("contrastive_temp", 0.3)

        # Orthogonal node query init
        self.node_queries = nn.Parameter(self._init_orthogonal_queries(
            self.num_nodes, self.node_dim))

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
        #self.decoder = PointCloudDecoder(
        #    self.node_dim, self.num_nodes, self.num_points, 3)
        self.decoder = NodeWiseDecoder(
            self.node_dim, self.num_nodes, self.num_points, 3)


        # [new-4] Node-Aware Readout (replaces SequenceReadout)
        self.readout = NodeAwareReadout(
            node_dim=self.node_dim,
            out_dim=self.out_dim,
            num_semantic_queries=self.num_semantic_queries,
            num_heads=num_heads)

        # [new-3] Motion Semantic Head
        me_cfg = config.get("motion_encoder", {})
        motion_feat_dim = me_cfg.get("feat_dim", 512)
        self.motion_semantic_head = MotionSemanticHead(
            motion_dim=motion_feat_dim,
            out_dim=self.out_dim,
            num_queries=self.num_semantic_queries,
            num_heads=num_heads)

        # Projection heads (for global contrastive)
        self.radar_proj = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim), nn.LayerNorm(self.out_dim), nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim))
        self.motion_proj = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim), nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim))

        # Momentum queue
        queue_size = cfg.get("queue_size", 256)
        self.queue_size = queue_size
        self.register_buffer("motion_queue",
                             torch.randn(queue_size, self.out_dim))
        self.register_buffer("queue_ptr",
                             torch.zeros(1, dtype=torch.long))

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

    @torch.no_grad()
    def _enqueue(self, g_m):
        batch_size = g_m.shape[0]
        ptr = int(self.queue_ptr)
        space = self.queue_size - ptr
        n = min(batch_size, space)
        if n > 0:
            self.motion_queue[ptr:ptr + n] = g_m[:n].detach()
            self.queue_ptr[0] = (ptr + n) % self.queue_size

    def forward_sequence(self, Y, temporal_mask=None):
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

        # [new-4] Node-Aware Readout
        rd = self.readout(node_seq, temporal_mask)

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
            "phase_logits": phase_logits,
            "kl": kl,
        }

    def forward(self, point_cloud, motion_seq=None, motion_mask=None,
                temporal_mask=None, phase_labels=None,
                phase_confidence=None):
        """
        Args:
            point_cloud: (B, T, N, D_in)
            motion_seq: (B, T', D_m) — motion encoder output sequence
            motion_mask: (B, T') bool — True=valid
            temporal_mask: (B, T) bool — radar temporal mask
        """
        out = self.forward_sequence(point_cloud, temporal_mask)
        device = point_cloud.device

        loss_obs = self._centered_chamfer(
            out["recon_sequence"], point_cloud[..., :3], temporal_mask)
        loss_kl = out["kl"]

        # ── Diversity (optional) ──
        if self.lambda_diversity > 0:
            metric_div = node_diversity_loss(out["node_history"], temporal_mask)
        else:
            with torch.no_grad():
                metric_div = node_diversity_loss(out["node_history"], temporal_mask)

        # ── [new-5,6] Dual contrastive: global + token ──
        loss_rm_global = torch.tensor(0.0, device=device)
        loss_rm_token = torch.tensor(0.0, device=device)

        if motion_seq is not None:
            motion_tokens, g_motion = self.motion_semantic_head(
                motion_seq, motion_mask)

            g_r = self.radar_proj(out["g_radar"])
            g_m = self.motion_proj(g_motion)

            loss_rm_global = global_contrastive_loss(
                g_r, g_m, temp=self.contrastive_temp)
            loss_rm_token = token_contrastive_loss(
                out["radar_tokens"], motion_tokens,
                temp=self.contrastive_temp)

            if self.training:
                self._enqueue(g_m.detach())

        # ── Phase ──
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

        # ── Total ──
        loss = (loss_obs
                + self.beta_kl * loss_kl
                + self.lambda_motion * loss_rm_global
                + self.lambda_token * loss_rm_token
                + self.lambda_phase * loss_phase
                + self.lambda_diversity * metric_div)

        return {
            "loss": loss,
            "loss_obs": loss_obs,
            "loss_kl": loss_kl,
            "loss_rm_global": loss_rm_global,
            "loss_rm_token": loss_rm_token,
            "loss_phase": loss_phase,
            "metric_div": metric_div,
            "g_radar": out["g_radar"],
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

    @torch.no_grad()
    def encode(self, point_cloud, temporal_mask=None):
        """Global radar embedding for evaluation."""
        return self.forward_sequence(point_cloud, temporal_mask)["g_radar"]

    @torch.no_grad()
    def encode_tokens(self, point_cloud, temporal_mask=None):
        """Semantic tokens for token-level evaluation."""
        out = self.forward_sequence(point_cloud, temporal_mask)
        return out["radar_tokens"], out["g_radar"]

    def get_diagnostics(self, model_output, temporal_mask=None):
        return self.diagnostics.compute(model_output, temporal_mask)