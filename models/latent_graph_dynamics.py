"""
Latent Graph Dynamics — v6.0 (Node Differentiation Fix)

v5.2 → v6.0 changes:
  [fix-1] ContextConditionedTransition: node-specific FiLM (gamma/beta per node)
  [fix-2] Node query: orthogonal initialization + larger scale
  [fix-3] LatentNodeInitializer: iterative slot competition (3 rounds)
  [fix-4] EvidenceExtractor: node-specific bias to break symmetry
  [fix-5] node_diversity_loss: diagnostic metric (NOT in loss — 구조적 fix만으로 분화 유도)
  [fix-6] Momentum queue for contrastive learning (small batch survival)
  [fix-7] NodeDiagnostics: training-time node differentiation monitor

Design principle:
  노드 분화는 loss로 강제하지 않고, 구조적 조건(fix 1~4)을 만들어
  기존 loss(L_obs, L_KL, L_rm)가 자연스럽게 분화를 유도하게 함.
  - L_obs: 노드가 각자 다른 신체 부위를 담당하면 reconstruction이 좋아짐
  - L_KL: 노드별 다른 prior/posterior가 KL 효율적
  - L_rm: global embedding의 정보량이 다양한 노드에서 나옴

Loss: L_obs + β·L_KL + λ_m·L_rm  [+ λ_ph·L_phase]
Diagnostic: metric_div (node cosine similarity, not backpropagated)
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
# [fix-7] Node Diagnostics — 훈련 중 분화 모니터링
# ═══════════════════════════════════════════════════════════

class NodeDiagnostics:
    """
    훈련 중 호출하여 노드 분화 상태를 추적.
    
    사용법:
        diag = NodeDiagnostics()
        # 매 epoch 또는 N step마다:
        report = diag.compute(model_output, temporal_mask)
        diag.log(report, epoch, step)
    """

    @staticmethod
    @torch.no_grad()
    def compute(model_output: dict, temporal_mask=None) -> dict:
        """
        model.forward_sequence() 출력에서 노드 분화 지표 계산.
        
        Returns dict with:
          - node_cosine_sim: 평균 node 간 cosine similarity (0=완전분화, 1=동일)
          - node_std: node 간 표현 편차 (높을수록 분화)
          - per_node_conf_std: node별 confidence 분산 (높을수록 각 node가 다르게 반응)
          - node_role_entropy: 각 node의 attention 패턴 엔트로피
          - query_cosine_sim: 학습된 node_queries 간 유사도
        """
        node_seq = model_output["node_history"]       # (B, T, M, D)
        conf_seq = model_output["confidence"]         # (B, T, M, 1)
        B, T, M, D = node_seq.shape

        # 1. Node cosine similarity (낮을수록 좋음)
        # frame별 node 간 평균 cosine sim
        node_frame = node_seq.mean(dim=1)             # (B, M, D)
        node_norm = F.normalize(node_frame, dim=-1)   # (B, M, D)
        sim_matrix = torch.bmm(node_norm, node_norm.transpose(1, 2))  # (B, M, M)
        # 대각 제외
        eye = torch.eye(M, device=sim_matrix.device).unsqueeze(0)
        off_diag = sim_matrix * (1 - eye)
        node_cos_sim = off_diag.sum() / (B * M * (M - 1))

        # 2. Node representation std (높을수록 좋음)
        node_std = node_frame.std(dim=1).mean()       # node 차원에서 std

        # 3. Per-node confidence variance
        # 각 frame에서 M개 node의 confidence가 얼마나 다른지
        conf_per_frame = conf_seq.squeeze(-1)          # (B, T, M)
        if temporal_mask is not None:
            valid = temporal_mask.unsqueeze(-1).float()  # (B, T, 1)
            conf_std = (conf_per_frame.std(dim=2) * valid.squeeze(-1)).sum() / valid.sum().clamp(1)
        else:
            conf_std = conf_per_frame.std(dim=2).mean()

        # 4. Node activation pattern (어떤 node가 가장 높은 confidence를 갖는지)
        # 이상적: 서로 다른 frame에서 서로 다른 node가 최고 confidence
        argmax_nodes = conf_per_frame.argmax(dim=2)   # (B, T)
        # 엔트로피: uniform이면 log(M), 하나에 집중이면 0
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
        """콘솔 출력."""
        cos = report["node_cosine_sim"]
        std = report["node_std"]
        cstd = report["conf_std_across_nodes"]
        ent = report["role_entropy_normalized"]

        # 판정
        if cos > 0.9:
            status = "⚠ COLLAPSED (nodes nearly identical)"
        elif cos > 0.7:
            status = "△ WEAK differentiation"
        elif cos > 0.4:
            status = "○ MODERATE differentiation"
        else:
            status = "★ GOOD differentiation"

        print(f"{prefix} ep{epoch} step{step}: "
              f"cos_sim={cos:.4f} node_std={std:.4f} "
              f"conf_std={cstd:.4f} role_ent={ent:.2f} → {status}")


# ═══════════════════════════════════════════════════════════
# Block 1. Observation Encoding (unchanged)
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
# Block 2. [fix-2,3] Latent Node Initialization — Slot Competition
# ═══════════════════════════════════════════════════════════

class LatentNodeInitializer(nn.Module):
    """
    True slot-competition initializer.

    핵심:
      1) slot -> point softmax
      2) point -> slot softmax
      3) top-k sparse masking
    을 직접 구현해서, 각 slot이 다른 point subset을 차지하도록 유도.
    """

    def __init__(
        self,
        node_dim,
        feat_dim,
        num_heads=4,
        num_rounds=3,
        top_k=16,
        compete_alpha=0.7,
    ):
        super().__init__()
        assert node_dim % num_heads == 0

        self.num_rounds = num_rounds
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.top_k = top_k
        self.compete_alpha = compete_alpha

        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(feat_dim, node_dim)
        self.v_proj = nn.Linear(feat_dim, node_dim)

        self.norm_slots = nn.LayerNorm(node_dim)
        self.norm_input = nn.LayerNorm(feat_dim)

        self.gru = nn.GRUCell(node_dim, node_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, node_dim * 2),
            nn.GELU(),
            nn.Linear(node_dim * 2, node_dim),
        )
        self.out_norm = nn.LayerNorm(node_dim)

    def forward(self, node_queries, point_features, point_mask=None):
        """
        Args:
            node_queries:   (B, M, D)
            point_features: (B, N, F)
            point_mask:     (B, N) bool
        Returns:
            slots:          (B, M, D)
        """
        B, M, D = node_queries.shape
        N = point_features.shape[1]
        H, d = self.num_heads, self.head_dim

        pf = self.norm_input(point_features)
        slots = node_queries

        if point_mask is not None:
            any_valid = point_mask.any(dim=-1)              # (B,)
            safe_mask = point_mask.clone()
            if (~any_valid).any():
                safe_mask[~any_valid, 0] = True
        else:
            any_valid = torch.ones(B, dtype=torch.bool, device=node_queries.device)
            safe_mask = torch.ones(B, N, dtype=torch.bool, device=node_queries.device)

        for _ in range(self.num_rounds):
            slots_norm = self.norm_slots(slots)

            Q = self.q_proj(slots_norm).view(B, M, H, d).transpose(1, 2)   # (B,H,M,d)
            K = self.k_proj(pf).view(B, N, H, d).transpose(1, 2)           # (B,H,N,d)
            V = self.v_proj(pf).view(B, N, H, d).transpose(1, 2)           # (B,H,N,d)

            logits = (Q @ K.transpose(-2, -1)) * self.scale                # (B,H,M,N)

            # invalid point masking
            vm = safe_mask[:, None, None, :].expand(B, H, M, N)
            logits = logits.masked_fill(~vm, float("-inf"))

            # top-k sparse masking on point axis
            k = min(self.top_k, N)
            topk_val, topk_idx = torch.topk(logits, k=k, dim=-1)           # (B,H,M,k)
            sparse_mask = torch.zeros_like(logits, dtype=torch.bool)
            sparse_mask.scatter_(-1, topk_idx, True)
            sparse_mask = sparse_mask & vm

            sparse_logits = logits.masked_fill(~sparse_mask, float("-inf"))

            # slot -> point attention
            attn_np = torch.softmax(sparse_logits, dim=-1)                 # (B,H,M,N)
            attn_np = torch.nan_to_num(attn_np, nan=0.0)

            # point -> slot competition
            attn_pn = torch.softmax(sparse_logits, dim=-2)                 # (B,H,M,N)
            attn_pn = torch.nan_to_num(attn_pn, nan=0.0)

            # combine
            attn = (1.0 - self.compete_alpha) * attn_np + self.compete_alpha * attn_pn

            # optional renorm over points for stability
            denom = attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            attn = attn / denom

            updates = attn @ V                                             # (B,H,M,d)
            updates = updates.transpose(1, 2).contiguous().view(B, M, D)

            # GRU update per slot
            slots = self.gru(
                updates.reshape(B * M, D),
                slots.reshape(B * M, D)
            ).view(B, M, D)

            # slot-wise FFN
            slots = slots + self.mlp(slots)
            slots = self.out_norm(slots)

            # if a sample had no valid point at all, keep original queries
            if (~any_valid).any():
                inv = (~any_valid)[:, None, None].float()
                slots = (1.0 - inv) * slots + inv * node_queries

        return slots


# ═══════════════════════════════════════════════════════════
# Block 3. Interaction Context Inference (unchanged)
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
            nn.Linear(256, ctx_dim),
        )
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
# Block 4. [fix-1] Context-Conditioned Transition — Node-Specific FiLM
# ═══════════════════════════════════════════════════════════

class ContextConditionedTransition(nn.Module):
    """
    Transition without node-mixing self-attention.
    Goal: keep node-specific trajectories separated.
    """

    def __init__(self, node_dim, num_nodes, ctx_dim=128, num_heads=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim

        # ctx -> node-specific FiLM parameters
        self.ctx_to_gamma_raw = nn.Linear(ctx_dim, num_nodes * node_dim)
        self.ctx_to_beta = nn.Linear(ctx_dim, num_nodes * node_dim)
        self.film_scale = 0.1

        self.pre_norm = nn.LayerNorm(node_dim)
        self.gru = nn.GRUCell(node_dim, node_dim)

        self.prior_mu = nn.Linear(node_dim, node_dim)
        self.prior_logvar = nn.Linear(node_dim, node_dim)

    def forward(self, prev_nodes, context):
        """
        prev_nodes: (B, M, D)
        context:    (B, C)
        """
        B, M, D = prev_nodes.shape

        nodes = self.pre_norm(prev_nodes)

        gamma_raw = self.ctx_to_gamma_raw(context).view(B, M, D)
        beta_raw = self.ctx_to_beta(context).view(B, M, D)

        gamma = 1.0 + self.film_scale * torch.tanh(gamma_raw)
        modulated = gamma * nodes + beta_raw

        prior_nodes = self.gru(
            modulated.reshape(B * M, D),
            prev_nodes.reshape(B * M, D)
        ).reshape(B, M, D)

        mu = self.prior_mu(prior_nodes)
        logvar = self.prior_logvar(prior_nodes).clamp(-6, 2)

        return prior_nodes, mu, logvar


# ═══════════════════════════════════════════════════════════
# Block 5. [fix-4] Evidence + Confidence — Node-Specific Bias
# ═══════════════════════════════════════════════════════════

class EvidenceExtractor(nn.Module):
    def __init__(self, node_dim, feat_dim, num_heads=4, compete_alpha=0.2, num_nodes=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        assert node_dim % num_heads == 0

        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(feat_dim, node_dim)
        self.v_proj = nn.Linear(feat_dim, node_dim)
        self.out_proj = nn.Linear(node_dim, node_dim)
        self.scale = self.head_dim ** -0.5

        # ★ node competition strength
        self.compete_alpha = compete_alpha

        # ★ [A] per-node query bias — symmetry breaking
        self.node_query_bias = nn.Parameter(
            torch.randn(1, num_nodes, node_dim) * 0.1)

        self.prior_for_conf = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.LayerNorm(node_dim // 2),
            nn.GELU(),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(num_heads * 2 + 1 + node_dim // 2, node_dim // 2),
            nn.GELU(),
            nn.Linear(node_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, prior_nodes, point_features, point_mask=None):
        B, M, _ = prior_nodes.shape
        N = point_features.shape[1]
        H, d = self.num_heads, self.head_dim

        Q = self.q_proj(prior_nodes + self.node_query_bias[:, :M, :]).view(B, M, H, d).transpose(1, 2)
        K = self.k_proj(point_features).view(B, N, H, d).transpose(1, 2)
        V = self.v_proj(point_features).view(B, N, H, d).transpose(1, 2)

        logits = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, M, N)
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

        # Confidence
        prior_conf = self.prior_for_conf(prior_nodes)

        conf_input = torch.cat([
            logit_max.permute(0, 2, 1),
            logit_mean.permute(0, 2, 1),
            n_valid.permute(0, 2, 1),
            prior_conf,
        ], dim=-1)
        confidence = self.confidence_head(conf_input)

        # ★ Dual softmax: node-to-point + point-to-node competition
        attn_np = torch.nan_to_num(logits.softmax(dim=-1), nan=0.0)  # (B,H,M,N) 기존
        if self.compete_alpha > 0:
            attn_pn = torch.nan_to_num(logits.softmax(dim=-2), nan=0.0)  # (B,H,M,N) node 경쟁
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
            nn.Linear(node_dim, node_dim),
        )
        self.fuse_norm = nn.LayerNorm(node_dim)
        self.post_mu = nn.Linear(node_dim, node_dim)
        self.post_logvar = nn.Linear(node_dim, node_dim)

    def forward(self, prior, evidence, confidence, training=True):
        fused = self.fuse(torch.cat([prior, evidence], dim=-1))
        coordinated = self.fuse_norm(confidence * fused + (1 - confidence) * prior)
        mu = self.post_mu(coordinated)
        logvar = self.post_logvar(coordinated).clamp(-6, 2)
        posterior = reparameterize(mu, logvar, training=training)
        return posterior, mu, logvar


# ═══════════════════════════════════════════════════════════
# Decoder (unchanged)
# ═══════════════════════════════════════════════════════════

class PointCloudDecoder(nn.Module):
    def __init__(self, node_dim, num_nodes, num_points=128, point_dim=3):
        super().__init__()
        self.num_points = num_points
        self.point_dim = point_dim
        self.decoder = nn.Sequential(
            nn.Linear(node_dim * num_nodes, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, num_points * point_dim),
        )

    def forward(self, nodes):
        B = nodes.shape[0]
        return self.decoder(nodes.reshape(B, -1)).reshape(
            B, self.num_points, self.point_dim)


# ═══════════════════════════════════════════════════════════
# Block 6. Sequence Readout (unchanged)
# ═══════════════════════════════════════════════════════════

class SequenceReadout(nn.Module):
    """
    Preserve node-wise tokens instead of collapsing nodes into mean/std too early.
    """

    def __init__(self, node_dim, num_nodes, out_dim=512, num_heads=4):
        super().__init__()
        self.node_dim = node_dim
        self.num_nodes = num_nodes

        # node token projection
        self.node_token_proj = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, node_dim),
            nn.GELU(),
        )

        # temporal attention over flattened (T * M) tokens
        self.temporal_pool = nn.MultiheadAttention(
            embed_dim=node_dim, num_heads=num_heads, batch_first=True
        )
        self.temporal_query = nn.Parameter(torch.randn(1, 1, node_dim) * 0.02)

        self.proj = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, out_dim)
        )

    def forward(self, node_seq, temporal_mask=None):
        """
        node_seq: (B, T, M, D)
        temporal_mask: (B, T) with True for valid steps
        """
        B, T, M, D = node_seq.shape

        tokens = self.node_token_proj(node_seq)   # (B,T,M,D)
        tokens = tokens.reshape(B, T * M, D)      # (B,TM,D)

        if temporal_mask is not None:
            # expand frame-valid mask to node-token-valid mask
            token_mask = temporal_mask.unsqueeze(-1).expand(B, T, M).reshape(B, T * M)
            key_padding_mask = ~token_mask
        else:
            key_padding_mask = None

        q = self.temporal_query.expand(B, -1, -1)
        seq_emb, _ = self.temporal_pool(
            q, tokens, tokens, key_padding_mask=key_padding_mask
        )

        return self.proj(seq_emb.squeeze(1))


# ═══════════════════════════════════════════════════════════
# Auxiliary: Phase Head (unchanged)
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
# [fix-5] Node Diversity Loss
# ═══════════════════════════════════════════════════════════

def node_diversity_loss(node_seq, temporal_mask=None):
    """
    Node 간 cosine similarity 측정 — 진단 전용 지표.
    
    이 함수는 loss에 포함되지 않음. 구조적 fix(1~4)가 기존 loss를 통해
    분화를 자연스럽게 유도하는지 모니터링하기 위한 metric.
    
    Args:
        node_seq: (B, T, M, D)
        temporal_mask: (B, T) bool
    Returns:
        scalar (0=완전 직교, 1=동일) — torch.no_grad()로 호출할 것
    """
    B, T, M, D = node_seq.shape
    node_norm = F.normalize(node_seq, dim=-1)     # (B, T, M, D)

    # (B, T, M, M) cosine similarity
    sim = torch.bmm(
        node_norm.reshape(B * T, M, D),
        node_norm.reshape(B * T, M, D).transpose(1, 2)
    ).reshape(B, T, M, M)

    # 대각 제거 (자기 자신과의 유사도 제외)
    eye = torch.eye(M, device=sim.device).view(1, 1, M, M)
    off_diag = (sim * (1 - eye)).pow(2)  # squared to penalize high sim

    if temporal_mask is not None:
        mask = temporal_mask.float().view(B, T, 1, 1)
        loss = (off_diag * mask).sum() / (mask.sum() * M * (M - 1)).clamp(1)
    else:
        loss = off_diag.sum() / (B * T * M * (M - 1))

    return loss


# ═══════════════════════════════════════════════════════════
# Full Model — v6.0
# ═══════════════════════════════════════════════════════════

class LatentGraphDynamicsModel(nn.Module):
    """
    v6.0: Node Differentiation Fix — structural changes only.
    
    Changes from v5.2:
      - [fix-1] Node-specific FiLM in transition (breaks gamma/beta broadcast)
      - [fix-2] Orthogonal node query init (nodes start different)
      - [fix-3] Iterative slot competition init (nodes attend different points)
      - [fix-4] Node-specific query bias in evidence (breaks Q symmetry)
      - [fix-6] Momentum queue for contrastive (small batch survival)
      - [fix-7] NodeDiagnostics (training-time monitoring)
    
    Loss is UNCHANGED: L_obs + β·L_KL + λ_m·L_rm [+ λ_ph·L_phase]
    Node diversity is a diagnostic metric, not a loss term.
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

        self.beta_kl = cfg.get("beta_kl", 0.1)
        self.lambda_motion = cfg.get("lambda_motion", 1.0)
        self.lambda_phase = cfg.get("lambda_phase", 0.0)
        self.contrastive_temp = cfg.get("contrastive_temp", 0.07)
        self.lambda_diversity = cfg.get("lambda_diversity", 0.1)

        # [fix-2] Orthogonal node query initialization
        self.node_queries = nn.Parameter(self._init_orthogonal_queries(
            self.num_nodes, self.node_dim))

        self.obs_encoder = ObservationEncoder(
            in_dim=self.point_in_dim,
            hidden_dims=cfg.get("encoder_hidden", [64, 128]),
            out_dim=self.feat_dim)

        # [fix-3] Slot competition initializer
        self.node_init = LatentNodeInitializer(
            node_dim=self.node_dim,
            feat_dim=self.feat_dim,
            num_heads=num_heads,
            num_rounds=cfg.get("init_rounds", 3),
            top_k=cfg.get("init_top_k", 16),
            compete_alpha=cfg.get("compete_alpha", 0.7),
        )

        self.context_encoder = InteractionContextEncoder(
            self.node_dim, self.num_nodes,
            ctx_dim=self.ctx_dim, history_len=self.ctx_history_len)

        # [fix-1] Node-specific FiLM transition
        self.transition = ContextConditionedTransition(
            self.node_dim, self.num_nodes, self.ctx_dim, num_heads)

        # [fix-4] Node-specific evidence extractor
        self.evidence_extractor = EvidenceExtractor(
            node_dim=self.node_dim,
            feat_dim=self.feat_dim,
            num_heads=num_heads,
            compete_alpha=cfg.get("compete_alpha", 0.2),
            num_nodes=self.num_nodes,
        )

        self.posterior_update = PosteriorUpdate(self.node_dim, num_heads)
        self.decoder = PointCloudDecoder(
            self.node_dim, self.num_nodes, self.num_points, 3)
        self.readout = SequenceReadout(
            self.node_dim, self.num_nodes, self.out_dim, num_heads)

        self.radar_proj = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim), nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim))
        self.motion_proj = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim), nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim))

        # [fix-6] Momentum queue for contrastive learning
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

        # [fix-7] Diagnostics helper
        self.diagnostics = NodeDiagnostics()

    @staticmethod
    def _init_orthogonal_queries(num_nodes, node_dim):
        """
        [fix-2] 직교 초기화: node query들이 처음부터 서로 다른 방향을 가리키도록.
        """
        if num_nodes <= node_dim:
            # QR decomposition으로 직교 벡터 생성
            rand = torch.randn(node_dim, num_nodes)
            q, _ = torch.linalg.qr(rand)
            queries = q[:, :num_nodes].t().unsqueeze(0)  # (1, M, D)
        else:
            # M > D인 경우: 랜덤 + 큰 스케일
            queries = torch.randn(1, num_nodes, node_dim) * 0.5
        return queries

    def _point_mask(self, Y):
        return Y[..., :3].norm(dim=-1) > 1e-6

    @torch.no_grad()
    def _enqueue(self, g_m):
        """[fix-6] Momentum queue에 motion embedding 추가."""
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
            # prefix-valid enforcement
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

        # [fix-2,3] Orthogonal queries + slot competition init
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
                prior_nodes, prior_mu, prior_lv = self.transition(
                    nodes, c_t)
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

        g_radar = self.readout(node_seq, temporal_mask)

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
            "g_radar": g_radar,
            "phase_logits": phase_logits,
            "kl": kl,
        }

    def forward(self, point_cloud, motion_features=None,
                temporal_mask=None, phase_labels=None,
                phase_confidence=None):
        out = self.forward_sequence(point_cloud, temporal_mask)
        device = point_cloud.device

        loss_obs = self._centered_chamfer(
            out["recon_sequence"], point_cloud[..., :3], temporal_mask)
        loss_kl = out["kl"]

        # Node diversity: diagnostic only, NOT in loss
        # 구조적 fix(1~4)가 기존 loss를 통해 자연스러운 분화를 유도함
        # Node diversity loss
        if self.lambda_diversity > 0:
            metric_div = node_diversity_loss(out["node_history"], temporal_mask)
        else:
            with torch.no_grad():
                metric_div = node_diversity_loss(out["node_history"], temporal_mask)

        # [fix-6] Contrastive with momentum queue
        loss_rm = torch.tensor(0.0, device=device)
        if motion_features is not None:
            g_r = self.radar_proj(out["g_radar"])
            g_m = self.motion_proj(motion_features)
            loss_rm = self._contrastive(g_r, g_m)

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

        # 원래 목적함수만 사용 — 구조적 fix가 분화를 자연스럽게 유도
        loss = (loss_obs
                + self.beta_kl * loss_kl
                + self.lambda_motion * loss_rm
                + self.lambda_phase * loss_phase
                + self.lambda_diversity * metric_div)

        return {
            "loss": loss,
            "loss_obs": loss_obs,
            "loss_kl": loss_kl,
            "loss_rm": loss_rm,
            "loss_phase": loss_phase,
            "metric_div": metric_div,     # diagnostic only, not in loss
            "g_radar": out["g_radar"],
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

    def _contrastive(self, g_r, g_m):
        """[fix-6] Momentum queue contrastive."""
        g_r = F.normalize(g_r, dim=-1)
        g_m = F.normalize(g_m, dim=-1)
        B = g_r.shape[0]

        # Current batch + queue negatives
        if self.training and self.queue_ptr > 0:
            queue_size = min(int(self.queue_ptr), self.queue_size)
            queue = F.normalize(self.motion_queue[:queue_size], dim=-1)
            g_m_all = torch.cat([g_m, queue], dim=0)
        else:
            g_m_all = g_m

        sim = g_r @ g_m_all.t() / self.contrastive_temp
        lab = torch.arange(B, device=sim.device)
        loss = F.cross_entropy(sim, lab)

        # Enqueue
        if self.training:
            self._enqueue(g_m)

        # Normalize by effective negatives
        #N_eff = max(g_m_all.shape[0], 2)
        return loss #/ max(np.log(N_eff), 1.0)

    @torch.no_grad()
    def encode(self, point_cloud, temporal_mask=None):
        return self.forward_sequence(point_cloud, temporal_mask)["g_radar"]

    def get_diagnostics(self, model_output, temporal_mask=None):
        """[fix-7] 외부에서 호출할 수 있는 진단 인터페이스."""
        return self.diagnostics.compute(model_output, temporal_mask)