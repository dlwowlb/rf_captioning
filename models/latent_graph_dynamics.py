"""
Latent Graph Dynamics — v5.2.
Context-Conditioned Variational Latent Dynamics for Radar Point Clouds.

v5.1 → v5.2 fixes:
  [fix-A] KL: per-sample 계산 + valid_t masking (invalid sample KL 배제)
  [fix-B] confidence: invalid timestep에서 이전 값 유지
  [fix-C] t=0: "prefix-valid" 가정 명시 + safety comment
  [fix-D] InteractionContextEncoder: empty defense에 batch_size 전달
  [fix-E] History buffer 단순화: grad buffer만 유지, detach buffer 제거
  [fix-F] EvidenceExtractor: all-invalid point → confidence=0, evidence=0

Loss: L_obs + β·L_KL + λ_m·L_rm  [+ λ_ph·L_phase]
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ═══════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════

def reparameterize(mu, logvar, training=True):
    if not training:
        return mu
    return mu + (0.5 * logvar).exp() * torch.randn_like(mu)


def kl_divergence_per_sample(mu_q, lv_q, mu_p, lv_p):
    """
    [fix-A] Per-sample KL divergence.
    Returns: (B,) — mean over (M, D) dims, NOT over batch.
    """
    var_q = lv_q.exp()
    var_p = lv_p.exp().clamp(min=1e-8)
    kl = 0.5 * (lv_p - lv_q + var_q / var_p
                 + (mu_q - mu_p).pow(2) / var_p - 1)
    # Mean over node and dim, keep batch
    return kl.mean(dim=list(range(1, kl.dim())))  # (B,)


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
# Block 2. Latent Node Initialization
# ═══════════════════════════════════════════════════════════

class LatentNodeInitializer(nn.Module):
    def __init__(self, node_dim, feat_dim, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=node_dim, num_heads=num_heads,
            kdim=feat_dim, vdim=feat_dim, batch_first=True)
        self.proj = nn.Linear(node_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, node_queries, point_features, point_mask=None):
        if point_mask is not None:
            # ★ all-invalid sample 방어: 해당 sample은 diagonal visible
            any_valid = point_mask.any(dim=-1)                    # (B,)
            safe_mask = point_mask.clone()
            if (~any_valid).any():
                safe_mask[~any_valid, 0] = True                   # 최소 1개 visible
            key_padding_mask = ~safe_mask
        else:
            key_padding_mask = None
        out, _ = self.cross_attn(
            node_queries, point_features, point_features,
            key_padding_mask=key_padding_mask)
        return self.norm(self.proj(out) + node_queries)


# ═══════════════════════════════════════════════════════════
# Block 3. Interaction Context Inference
# ═══════════════════════════════════════════════════════════

class InteractionContextEncoder(nn.Module):
    """
    [fix-D] batch_size를 forward에서 직접 받지 않고,
    빈 리스트는 caller에서 오지 않도록 보장.
    assert로 방어.
    """

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
        """
        node_history: list of (B, M, D), length >= 1.
        Returns: c_t (B, ctx_dim)
        """
        assert len(node_history) >= 1, \
            "Context encoder requires at least 1 history frame"

        B = node_history[0].shape[0]
        device = node_history[0].device
        K = self.history_len

        # Take at most K, zero-pad front if shorter
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
# Block 4. Context-Conditioned Variational Transition
# ═══════════════════════════════════════════════════════════

class ContextConditionedTransition(nn.Module):
    """FiLM-conditioned transition with bounded gamma."""

    def __init__(self, node_dim, ctx_dim=128, num_heads=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=node_dim, num_heads=num_heads, batch_first=True)
        self.sa_norm = nn.LayerNorm(node_dim)
        self.pre_sa_norm = nn.LayerNorm(node_dim)

        self.ctx_to_gamma_raw = nn.Linear(ctx_dim, node_dim)
        self.ctx_to_beta = nn.Linear(ctx_dim, node_dim)
        self.film_scale = 0.1

        self.gru = nn.GRUCell(node_dim, node_dim)
        self.prior_mu = nn.Linear(node_dim, node_dim)
        self.prior_logvar = nn.Linear(node_dim, node_dim)

    def forward(self, prev_nodes, context):
        B, M, D = prev_nodes.shape
        normed = self.pre_sa_norm(prev_nodes)
        sa_out, _ = self.self_attn(normed, normed, normed)
        nodes = self.sa_norm(prev_nodes + sa_out)

        gamma = 1.0 + self.film_scale * torch.tanh(
            self.ctx_to_gamma_raw(context)).unsqueeze(1)
        beta = self.ctx_to_beta(context).unsqueeze(1)
        nodes = gamma * nodes + beta

        prior_nodes = self.gru(
            nodes.reshape(B * M, D),
            prev_nodes.reshape(B * M, D)).reshape(B, M, D)

        mu = self.prior_mu(prior_nodes)
        logvar = self.prior_logvar(prior_nodes).clamp(-6, 2)
        return prior_nodes, mu, logvar


# ═══════════════════════════════════════════════════════════
# Block 5. Evidence + Attention-Derived Confidence
# ═══════════════════════════════════════════════════════════

class EvidenceExtractor(nn.Module):
    """
    [fix-F] All-invalid point 처리: confidence=0, evidence=0 강제.
    """

    def __init__(self, node_dim, feat_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        assert node_dim % num_heads == 0

        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(feat_dim, node_dim)
        self.v_proj = nn.Linear(feat_dim, node_dim)
        self.out_proj = nn.Linear(node_dim, node_dim)
        self.scale = self.head_dim ** -0.5

        self.confidence_head = nn.Sequential(
            nn.Linear(num_heads * 2 + 1, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, prior_nodes, point_features, point_mask=None):
        B, M, _ = prior_nodes.shape
        N = point_features.shape[1]
        H, d = self.num_heads, self.head_dim

        Q = self.q_proj(prior_nodes).view(B, M, H, d).transpose(1, 2)
        K = self.k_proj(point_features).view(B, N, H, d).transpose(1, 2)
        V = self.v_proj(point_features).view(B, N, H, d).transpose(1, 2)

        logits = (Q @ K.transpose(-2, -1)) * self.scale

        if point_mask is not None:
            vm = point_mask[:, None, None, :].expand_as(logits)
            any_valid = point_mask.any(dim=-1)               # (B,)
            has_invalid = (~any_valid).any()

            # Masked stats: only valid points
            logits_masked = logits.masked_fill(~vm, float("-inf"))
            logit_max_raw = logits_masked.max(dim=-1).values  # (B, H, M)

            logits_zeroed = logits * vm.float()
            valid_count = vm.float().sum(dim=-1).clamp(min=1)
            logit_mean_raw = logits_zeroed.sum(dim=-1) / valid_count

            n_valid_ratio = point_mask.float().sum(dim=-1) / N  # (B,)
            n_valid = n_valid_ratio.view(B, 1, 1).expand(B, 1, M)

            # All-invalid samples: zero out stats cleanly
            # (so confidence_head sees clean zeros, not -inf artifacts)
            if has_invalid:
                inv = (~any_valid)[:, None, None].expand_as(logit_max_raw)
                logit_max_raw = logit_max_raw.masked_fill(inv, 0.0)
                logit_mean_raw = logit_mean_raw.masked_fill(inv, 0.0)
                # n_valid is already 0 for these samples

            logit_max = logit_max_raw
            logit_mean = logit_mean_raw

            # Mask logits for softmax attention
            logits = logits.masked_fill(~vm, float("-inf"))
        else:
            any_valid = torch.ones(B, dtype=torch.bool, device=prior_nodes.device)
            has_invalid = False
            logit_max = logits.max(dim=-1).values
            logit_mean = logits.mean(dim=-1)
            n_valid = torch.ones(B, 1, M, device=prior_nodes.device)

        # Confidence from logit stats
        conf_input = torch.cat([
            logit_max.permute(0, 2, 1),                      # (B, M, H)
            logit_mean.permute(0, 2, 1),                     # (B, M, H)
            n_valid.permute(0, 2, 1),                        # (B, M, 1)
        ], dim=-1)
        confidence = self.confidence_head(conf_input)        # (B, M, 1)

        # Evidence via softmax attention
        logits = logits.clamp(-30, 30)
        attn = torch.nan_to_num(logits.softmax(dim=-1), nan=0.0)
        out = (attn @ V).transpose(1, 2).reshape(B, M, -1)
        evidence = self.out_proj(out)

        # All-invalid: force zero (safety, stats are already clean)
        if has_invalid:
            mask_f = (~any_valid)[:, None, None].float()
            confidence = confidence * (1 - mask_f)
            evidence = evidence * (1 - mask_f)

        return evidence, confidence


class PosteriorUpdate(nn.Module):
    """Single-pass: fuse → confidence gate → self-attn → sample."""

    def __init__(self, node_dim, num_heads=4):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim), nn.GELU(),
            nn.Linear(node_dim, node_dim),
        )
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
            nn.Linear(512, num_points * point_dim),
        )

    def forward(self, nodes):
        B = nodes.shape[0]
        return self.decoder(nodes.reshape(B, -1)).reshape(
            B, self.num_points, self.point_dim)


# ═══════════════════════════════════════════════════════════
# Block 6. Sequence Readout
# ═══════════════════════════════════════════════════════════

class SequenceReadout(nn.Module):
    def __init__(self, node_dim, num_nodes, out_dim=512, num_heads=4):
        super().__init__()
        self.node_pool = nn.MultiheadAttention(
            embed_dim=node_dim, num_heads=num_heads, batch_first=True)
        self.node_query = nn.Parameter(torch.randn(1, 1, node_dim) * 0.02)
        self.temporal_pool = nn.MultiheadAttention(
            embed_dim=node_dim, num_heads=num_heads, batch_first=True)
        self.temporal_query = nn.Parameter(torch.randn(1, 1, node_dim) * 0.02)
        self.proj = nn.Sequential(
            nn.LayerNorm(node_dim), nn.Linear(node_dim, out_dim))

    def forward(self, node_seq, temporal_mask=None):
        B, T, M, D = node_seq.shape
        nodes_flat = node_seq.reshape(B * T, M, D)
        q = self.node_query.expand(B * T, -1, -1)
        frame_emb, _ = self.node_pool(q, nodes_flat, nodes_flat)
        frame_emb = frame_emb.squeeze(1).reshape(B, T, D)
        q_t = self.temporal_query.expand(B, -1, -1)
        kp = ~temporal_mask if temporal_mask is not None else None
        seq_emb, _ = self.temporal_pool(
            q_t, frame_emb, frame_emb, key_padding_mask=kp)
        return self.proj(seq_emb.squeeze(1))


# ═══════════════════════════════════════════════════════════
# Auxiliary: Phase Head (optional)
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
# Full Model
# ═══════════════════════════════════════════════════════════

class LatentGraphDynamicsModel(nn.Module):
    """
    Context-Conditioned Variational Latent Dynamics — v5.2.

    Assumptions:
      [fix-C] 데이터는 prefix-valid 구조를 가정.
      즉 각 샘플의 유효 프레임은 t=0부터 시작하여 연속적이고,
      padding은 뒤에만 붙음. t=0이 invalid인 경우는 없다고 가정.
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

        self.node_queries = nn.Parameter(
            torch.randn(1, self.num_nodes, self.node_dim) * 0.02)

        self.obs_encoder = ObservationEncoder(
            in_dim=self.point_in_dim,
            hidden_dims=cfg.get("encoder_hidden", [64, 128]),
            out_dim=self.feat_dim)
        self.node_init = LatentNodeInitializer(
            self.node_dim, self.feat_dim, num_heads)
        self.context_encoder = InteractionContextEncoder(
            self.node_dim, self.num_nodes,
            ctx_dim=self.ctx_dim, history_len=self.ctx_history_len)
        self.transition = ContextConditionedTransition(
            self.node_dim, self.ctx_dim, num_heads)
        self.evidence_extractor = EvidenceExtractor(
            self.node_dim, self.feat_dim, num_heads)
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

        if self.lambda_phase > 0:
            self.phase_head = PhaseHead(self.node_dim, self.num_nodes)
        else:
            self.phase_head = None

    def _point_mask(self, Y):
        return Y[..., :3].norm(dim=-1) > 1e-6

    def forward_sequence(self, Y, temporal_mask=None):
        B, T, N, _ = Y.shape
        device = Y.device
        is_train = self.training

        # Prefix-valid assertion: valid frames contiguous from t=0
        if temporal_mask is not None:
            assert temporal_mask[:, 0].all(), \
                "prefix-valid violated: t=0 must be valid for all samples"
            # valid→invalid transition은 단조감소여야 함 (1110 OK, 1010 NG)
            assert (temporal_mask[:, 1:] <= temporal_mask[:, :-1]).all(), \
                "prefix-valid violated: valid frames must be contiguous from t=0"

        U = self.obs_encoder(Y)
        pmask = self._point_mask(Y)

        # ★ temporal_mask과 point_mask 동기화:
        #   프레임이 valid라고 되어있어도 실제 유효 포인트가 0이면 invalid 처리
        if temporal_mask is not None:
            frame_has_points = pmask.any(dim=-1)               # (B, T) — 프레임에 유효 점이 있는지
            temporal_mask = temporal_mask & frame_has_points    # 둘 다 True여야 valid

            # prefix-valid 재보장: 첫 invalid 이후는 전부 invalid
            for b in range(B):
                valid_len = temporal_mask[b].long().sum()
                temporal_mask[b, valid_len:] = False

            # t=0이 invalid이면 강제로 살림 (최소 1프레임 보장)
            if not temporal_mask[:, 0].all():
                temporal_mask[:, 0] = True



        # [fix-C] t=0은 항상 valid라고 가정 (prefix-valid 구조)
        queries = self.node_queries.expand(B, -1, -1)
        nodes = self.node_init(queries, U[:, 0], pmask[:, 0])

        # [fix-E] 단순화: grad buffer만 유지 (최근 K개)
        # K를 초과하면 oldest를 detach 후 제거
        node_hist = []  # with gradient, max K entries

        # [fix-B] 이전 confidence (invalid frame에서 유지할 값)
        prev_conf = torch.zeros(B, self.num_nodes, 1, device=device)

        H_node, H_recon, H_conf = [], [], []
        H_context, H_post_lv = [], []                        # diagnostics
        kl_sum = torch.tensor(0.0, device=device)
        kl_weight = torch.tensor(0.0, device=device)

        for t in range(T):
            U_t = U[:, t]
            m_t = pmask[:, t]

            if temporal_mask is not None:
                valid_t = temporal_mask[:, t]                  # (B,)
            else:
                valid_t = torch.ones(B, dtype=torch.bool, device=device)

            prev_nodes = nodes

            if t == 0:
                ev, conf = self.evidence_extractor(nodes, U_t, m_t)
                new_nodes, post_mu, post_lv = self.posterior_update(
                    nodes, ev, conf, training=is_train)
                # No KL at t=0; no context at t=0
                c_t = torch.zeros(B, self.ctx_dim, device=device)

            else:
                # Context from recent history (gradient flows through)
                c_t = self.context_encoder(node_hist[-self.ctx_history_len:])

                prior_nodes, prior_mu, prior_lv = self.transition(
                    nodes, c_t)

                ev, conf = self.evidence_extractor(prior_nodes, U_t, m_t)
                new_nodes, post_mu, post_lv = self.posterior_update(
                    prior_nodes, ev, conf, training=is_train)

                # [fix-A] Per-sample KL, global valid-sample average
                kl_per_sample = kl_divergence_per_sample(
                    post_mu, post_lv, prior_mu, prior_lv)     # (B,)
                kl_sum = kl_sum + (kl_per_sample * valid_t.float()).sum()
                kl_weight = kl_weight + valid_t.float().sum()

            # State masking: invalid → keep previous
            valid_f = valid_t[:, None, None].float()
            nodes = valid_f * new_nodes + (1 - valid_f) * prev_nodes

            # [fix-B] Confidence masking: invalid → keep previous
            conf = valid_f * conf + (1 - valid_f) * prev_conf
            prev_conf = conf.detach()

            recon = self.decoder(nodes)

      



            # [fix-E] History: keep with grad, trim to K
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
        ctx_seq = torch.stack(H_context, 1)                  # (B, T, ctx_dim)
        post_lv_seq = torch.stack(H_post_lv, 1)              # (B, T, M, D)

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

            # Temporal mask: only compute phase loss on valid frames
            if temporal_mask is not None:
                pm = temporal_mask[:, :Tm].reshape(-1)        # (B*Tm,) bool
                lf = lf[pm]
                lb = lb[pm]
            else:
                pm = None

            if lf.shape[0] > 0:  # at least one valid frame
                if phase_confidence is not None:
                    w = phase_confidence[:, :Tm].reshape(-1).to(device)
                    if pm is not None:
                        w = w[pm]
                    loss_phase = (F.cross_entropy(
                        lf, lb, reduction="none") * w).mean()
                else:
                    loss_phase = F.cross_entropy(lf, lb)

        loss = (loss_obs
                + self.beta_kl * loss_kl
                + self.lambda_motion * loss_rm
                + self.lambda_phase * loss_phase)

        return {
            "loss": loss,
            "loss_obs": loss_obs,
            "loss_kl": loss_kl,
            "loss_rm": loss_rm,
            "loss_phase": loss_phase,
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
        g_r = F.normalize(g_r, dim=-1)
        g_m = F.normalize(g_m, dim=-1)
        sim = g_r @ g_m.t() / self.contrastive_temp
        lab = torch.arange(sim.shape[0], device=sim.device)
        B = sim.shape[0]
        raw = (F.cross_entropy(sim, lab) + F.cross_entropy(sim.t(), lab)) / 2
        return raw / max(np.log(B), 1.0)   # ★ log(B)로 나눠서 정규화
        '''
        g_r = F.normalize(g_r, dim=-1)
        g_m = F.normalize(g_m, dim=-1)
        sim = g_r @ g_m.t() / self.contrastive_temp
        lab = torch.arange(sim.shape[0], device=sim.device)
        return (F.cross_entropy(sim, lab) + F.cross_entropy(sim.t(), lab)) / 2
        '''

    @torch.no_grad()
    def encode(self, point_cloud, temporal_mask=None):
        return self.forward_sequence(point_cloud, temporal_mask)["g_radar"]