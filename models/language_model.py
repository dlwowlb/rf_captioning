"""
Radar-Aware Language Model for RadarLLM (Section 4.2).

Built on T5 with:
- Extended vocabulary: V = V_text ∪ V_radar (32768 WordPieces + K radar tokens + <som>/<eom>)
- Shared 512-dim embeddings
- Multi-task pre-training: radar prediction, radar→text, text→radar
- Instruction tuning with similarity-based loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config


class RadarAwareLanguageModel(nn.Module):
    """
    Radar-aware language model that translates radar tokens to/from text.
    
    The model extends T5's vocabulary with K radar tokens and special
    start/end-of-motion tokens, then trains via:
    
    Pre-training:
      (1) Radar Prediction — masked radar token recovery (Eq. 7)
      (2) Radar→Text — encode radar, decode text (Eq. 8)
      (3) Text→Radar — encode text, decode radar (Eq. 9)
    
    Instruction Tuning:
      Fine-tune on prompted radar-to-text with similarity loss
    """

    def __init__(self, config: dict):
        super().__init__()
        lm_cfg = config.get("language_model", config)
        
        self.backbone_name = lm_cfg.get("backbone", "google/flan-t5-small")
        self.embed_dim = lm_cfg.get("embed_dim", 512)
        self.codebook_size = lm_cfg.get("num_radar_tokens", 512)

        # Load T5 backbone
        self.tokenizer = T5Tokenizer.from_pretrained(self.backbone_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.backbone_name)

        # Get original vocab size
        self.orig_vocab_size = len(self.tokenizer)

        # Add special tokens
        special_tokens = lm_cfg.get("special_tokens", ["<som>", "<eom>"])
        # Add radar tokens: <radar_0>, <radar_1>, ..., <radar_{K-1}>
        radar_tokens = [f"<radar_{i}>" for i in range(self.codebook_size)]
        all_new_tokens = special_tokens + radar_tokens

        num_added = self.tokenizer.add_tokens(all_new_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Token ID ranges
        self.som_token_id = self.tokenizer.convert_tokens_to_ids("<som>")
        self.eom_token_id = self.tokenizer.convert_tokens_to_ids("<eom>")
        self.radar_token_offset = self.tokenizer.convert_tokens_to_ids("<radar_0>")

        # Radar embedding projection (from codebook dim to T5 embed dim)
        t5_embed_dim = self.model.config.d_model
        if self.embed_dim != t5_embed_dim:
            self.radar_proj = nn.Linear(self.embed_dim, t5_embed_dim)
        else:
            self.radar_proj = nn.Identity()

        # Pre-training loss weights
        pretrain_cfg = lm_cfg.get("pretrain", {})
        self.lambda_pred = pretrain_cfg.get("lambda_pred", 1.0)
        self.lambda_r2t = pretrain_cfg.get("lambda_r2t", 1.0)
        self.lambda_t2r = pretrain_cfg.get("lambda_t2r", 1.0)
        self.mask_ratio = pretrain_cfg.get("mask_ratio", 0.15)

    def radar_indices_to_token_ids(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert VQ-VAE codebook indices to T5 token IDs."""
        return indices + self.radar_token_offset

    def token_ids_to_radar_indices(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convert T5 token IDs back to VQ-VAE codebook indices."""
        return token_ids - self.radar_token_offset

    def _build_radar_text_input(
        self,
        radar_indices: torch.Tensor,
        text_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build combined input sequence: [radar_tokens, text_tokens] (Eq. 6).
        
        Returns input_ids and attention_mask.
        """
        B, L = radar_indices.shape
        device = radar_indices.device

        # Convert radar indices to token IDs
        radar_ids = self.radar_indices_to_token_ids(radar_indices)

        # Add <som> and <eom>
        som = torch.full((B, 1), self.som_token_id, device=device, dtype=torch.long)
        eom = torch.full((B, 1), self.eom_token_id, device=device, dtype=torch.long)
        radar_seq = torch.cat([som, radar_ids, eom], dim=1)  # (B, L+2)
        radar_mask = torch.ones_like(radar_seq)

        if text_ids is not None:
            input_ids = torch.cat([radar_seq, text_ids], dim=1)
            attention_mask = torch.cat([radar_mask, text_attention_mask], dim=1)
        else:
            input_ids = radar_seq
            attention_mask = radar_mask

        return input_ids, attention_mask

    # ───────────────────────────────────────────────
    # Pre-training tasks
    # ───────────────────────────────────────────────

    def radar_prediction_loss(self, radar_indices: torch.Tensor) -> torch.Tensor:
        """
        Radar Prediction task (Eq. 7): mask 15% of radar tokens,
        predict originals via span corruption.
        """
        B, L = radar_indices.shape
        device = radar_indices.device

        radar_ids = self.radar_indices_to_token_ids(radar_indices)

        # Create mask
        num_mask = max(1, int(L * self.mask_ratio))
        mask_positions = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            idx = torch.randperm(L, device=device)[:num_mask]
            mask_positions[b, idx] = True

        # Sentinel token (T5 uses <extra_id_X> for masking)
        sentinel_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")

        # Replace masked positions with sentinel
        input_ids = radar_ids.clone()
        input_ids[mask_positions] = sentinel_id

        # Add <som>/<eom>
        som = torch.full((B, 1), self.som_token_id, device=device, dtype=torch.long)
        eom = torch.full((B, 1), self.eom_token_id, device=device, dtype=torch.long)
        encoder_input = torch.cat([som, input_ids, eom], dim=1)
        encoder_mask = torch.ones_like(encoder_input)

        # Target: the masked token IDs
        target_ids = radar_ids[mask_positions].reshape(B, num_mask)
        # Add sentinel prefix for T5 decoder format
        sentinel_prefix = torch.full((B, 1), sentinel_id, device=device, dtype=torch.long)
        decoder_labels = torch.cat([sentinel_prefix, target_ids], dim=1)

        outputs = self.model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            labels=decoder_labels,
        )
        return outputs.loss

    def radar_to_text_loss(
        self,
        radar_indices: torch.Tensor,
        text_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Radar→Text task (Eq. 8): encode radar tokens, decode text.
        """
        B = radar_indices.shape[0]
        device = radar_indices.device

        radar_ids = self.radar_indices_to_token_ids(radar_indices)
        som = torch.full((B, 1), self.som_token_id, device=device, dtype=torch.long)
        eom = torch.full((B, 1), self.eom_token_id, device=device, dtype=torch.long)
        encoder_input = torch.cat([som, radar_ids, eom], dim=1)
        encoder_mask = torch.ones_like(encoder_input)

        outputs = self.model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            labels=text_ids,
        )
        return outputs.loss

    def text_to_radar_loss(
        self,
        text_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        radar_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Text→Radar task (Eq. 9): encode text, decode radar tokens.
        """
        radar_ids = self.radar_indices_to_token_ids(radar_indices)
        B = text_ids.shape[0]
        device = text_ids.device

        som = torch.full((B, 1), self.som_token_id, device=device, dtype=torch.long)
        eom = torch.full((B, 1), self.eom_token_id, device=device, dtype=torch.long)
        decoder_labels = torch.cat([som, radar_ids, eom], dim=1)

        outputs = self.model(
            input_ids=text_ids,
            attention_mask=text_attention_mask,
            labels=decoder_labels,
        )
        return outputs.loss

    def pretrain_step(
        self,
        radar_indices: torch.Tensor,
        text_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Combined pre-training loss (Eq. 10):
        L_pretrain = λ1·L_pred + λ2·L_r2t + λ3·L_t2r
        """
        loss_pred = self.radar_prediction_loss(radar_indices)
        loss_r2t = self.radar_to_text_loss(radar_indices, text_ids, text_attention_mask)
        loss_t2r = self.text_to_radar_loss(text_ids, text_attention_mask, radar_indices)

        total = (
            self.lambda_pred * loss_pred
            + self.lambda_r2t * loss_r2t
            + self.lambda_t2r * loss_t2r
        )

        return {
            "loss": total,
            "loss_pred": loss_pred,
            "loss_r2t": loss_r2t,
            "loss_t2r": loss_t2r,
        }

    # ───────────────────────────────────────────────
    # Instruction tuning
    # ───────────────────────────────────────────────

    def instruction_tune_step(
        self,
        radar_indices: torch.Tensor,
        text_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        prompt_template: str = "Describe the motion <Motion Placeholder>.",
    ) -> Dict[str, torch.Tensor]:
        """
        Instruction-tuning stage: prompted radar-to-text generation.
        
        The prompt is tokenized and prepended to radar tokens as encoder input.
        """
        B = radar_indices.shape[0]
        device = radar_indices.device

        # Tokenize prompt
        prompt_encoding = self.tokenizer(
            [prompt_template] * B,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        # Build encoder input: [prompt_tokens, <som>, radar_tokens, <eom>]
        radar_ids = self.radar_indices_to_token_ids(radar_indices)
        som = torch.full((B, 1), self.som_token_id, device=device, dtype=torch.long)
        eom = torch.full((B, 1), self.eom_token_id, device=device, dtype=torch.long)
        radar_seq = torch.cat([som, radar_ids, eom], dim=1)

        encoder_input = torch.cat([prompt_encoding.input_ids, radar_seq], dim=1)
        encoder_mask = torch.cat(
            [prompt_encoding.attention_mask, torch.ones_like(radar_seq)], dim=1
        )

        outputs = self.model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            labels=text_ids,
        )
        return {"loss": outputs.loss}

    # ───────────────────────────────────────────────
    # Inference
    # ───────────────────────────────────────────────

    @torch.no_grad()
    def generate_text(
        self,
        radar_indices: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 5,
        prompt_template: Optional[str] = None,
    ) -> List[str]:
        """
        Generate text descriptions from radar token sequences.
        
        Args:
            radar_indices: (B, L) codebook indices
            max_length: max generation length
            num_beams: beam search width
            prompt_template: optional instruction prompt
        
        Returns:
            List of generated text descriptions
        """
        B = radar_indices.shape[0]
        device = radar_indices.device

        radar_ids = self.radar_indices_to_token_ids(radar_indices)
        som = torch.full((B, 1), self.som_token_id, device=device, dtype=torch.long)
        eom = torch.full((B, 1), self.eom_token_id, device=device, dtype=torch.long)
        radar_seq = torch.cat([som, radar_ids, eom], dim=1)

        if prompt_template:
            prompt_enc = self.tokenizer(
                [prompt_template] * B,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            encoder_input = torch.cat([prompt_enc.input_ids, radar_seq], dim=1)
            encoder_mask = torch.cat(
                [prompt_enc.attention_mask, torch.ones_like(radar_seq)], dim=1
            )
        else:
            encoder_input = radar_seq
            encoder_mask = torch.ones_like(radar_seq)

        outputs = self.model.generate(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

        texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return texts


class RadarLLM(nn.Module):
    """
    Full RadarLLM pipeline: VQ-VAE tokenizer + Radar-aware LM.
    
    point_cloud → AggregateVQVAE.encode() → token indices
    → RadarAwareLanguageModel.generate_text() → text description
    """

    def __init__(self, config: dict):
        super().__init__()
        from .aggregate_vqvae_backup import AggregateVQVAE

        self.tokenizer = AggregateVQVAE(config)
        self.language_model = RadarAwareLanguageModel(config)

    @torch.no_grad()
    def predict(
        self,
        point_cloud: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 5,
    ) -> List[str]:
        """
        End-to-end inference: radar point cloud → text description.
        """
        indices, _ = self.tokenizer.encode(point_cloud)
        texts = self.language_model.generate_text(
            indices,
            max_length=max_length,
            num_beams=num_beams,
            prompt_template="Describe the motion.",
        )
        return texts
