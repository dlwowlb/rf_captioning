"""
Radar-aware Language Model.

Integrates radar tokens with text tokens in a unified vocabulary,
using FLAN-T5-small as the backbone for cross-modal understanding
and auto-regressive caption generation.
"""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from typing import Dict, Optional


class RadarLanguageModel(nn.Module):
    """Radar-aware Language Model with shared embeddings.

    Architecture:
        1. Shared Embedding Layer: projects both radar tokens (from VQ codebook)
           and text tokens (WordPieces) into a unified 512-dim space.
        2. FLAN-T5-small backbone: 6 encoder + 6 decoder transformer layers
           for cross-modal attention and auto-regressive text generation.

    The model takes radar token IDs as encoder input and generates
    natural language descriptions auto-regressively from the decoder.
    """

    def __init__(self, radar_vocab_size: int = 512, embed_dim: int = 512,
                 text_model_name: str = "google/flan-t5-small",
                 max_radar_tokens: int = 16, max_text_len: int = 128,
                 label_smoothing: float = 0.1,
                 codebook_dim: int = 512):
        super().__init__()
        self.radar_vocab_size = radar_vocab_size
        self.embed_dim = embed_dim
        self.max_radar_tokens = max_radar_tokens
        self.max_text_len = max_text_len

        # Load T5 tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(text_model_name)
        self.text_vocab_size = self.tokenizer.vocab_size

        # Total vocab = text vocab + radar codebook tokens
        self.total_vocab_size = self.text_vocab_size + radar_vocab_size

        # Build T5 model with extended vocabulary
        t5_config = T5Config.from_pretrained(text_model_name)
        t5_config.vocab_size = self.total_vocab_size
        t5_config.d_model = embed_dim

        # Try loading pretrained weights, fall back to from-scratch
        try:
            self.t5 = T5ForConditionalGeneration.from_pretrained(
                text_model_name, config=t5_config, ignore_mismatched_sizes=True
            )
        except Exception:
            self.t5 = T5ForConditionalGeneration(t5_config)

        # Resize token embeddings to accommodate radar tokens
        self.t5.resize_token_embeddings(self.total_vocab_size)

        # Radar token projection: map VQ codebook vectors to embedding space
        self.radar_embed_proj = nn.Linear(codebook_dim, embed_dim)

        # Loss
        self.label_smoothing = label_smoothing

        # Special token IDs for radar prefix
        self.radar_token_offset = self.text_vocab_size  # radar tokens start after text vocab

    def get_radar_input_ids(self, radar_token_ids: torch.Tensor) -> torch.Tensor:
        """Convert radar VQ token IDs to unified vocabulary IDs.

        Args:
            radar_token_ids: (B, M) token IDs from VQ codebook [0, radar_vocab_size)

        Returns:
            unified_ids: (B, M) IDs shifted into unified vocabulary space
        """
        return radar_token_ids + self.radar_token_offset

    def forward(self, radar_token_ids: torch.Tensor,
                radar_features: Optional[torch.Tensor] = None,
                text_input_ids: Optional[torch.Tensor] = None,
                text_attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            radar_token_ids: (B, M) discrete radar token indices from VQ
            radar_features: (B, M, codebook_dim) optional continuous radar features
            text_input_ids: (B, L) tokenized caption input (for teacher forcing)
            text_attention_mask: (B, L) attention mask for text
            labels: (B, L) target token IDs for loss computation

        Returns:
            dict with:
                - loss: language modeling loss
                - logits: (B, L, vocab_size) output logits
        """
        B = radar_token_ids.shape[0]
        device = radar_token_ids.device

        # Convert radar tokens to unified vocab IDs
        encoder_input_ids = self.get_radar_input_ids(radar_token_ids)

        # Create attention mask for encoder (all ones for radar tokens)
        encoder_attention_mask = torch.ones_like(encoder_input_ids)

        # If continuous radar features provided, inject them as encoder embeddings
        if radar_features is not None:
            # Project radar features to T5 embedding space
            radar_embeds = self.radar_embed_proj(radar_features)  # (B, M, embed_dim)

            # Get the T5 encoder directly with custom embeddings
            encoder_outputs = self.t5.encoder(
                inputs_embeds=radar_embeds,
                attention_mask=encoder_attention_mask,
            )

            outputs = self.t5(
                encoder_outputs=encoder_outputs,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=text_input_ids,
                decoder_attention_mask=text_attention_mask,
                labels=labels,
            )
        else:
            # Use token IDs (discrete) through the embedding table
            outputs = self.t5(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=text_input_ids,
                decoder_attention_mask=text_attention_mask,
                labels=labels,
            )

        result = {"logits": outputs.logits}
        if labels is not None:
            result["loss"] = outputs.loss
        return result

    @torch.no_grad()
    def generate_caption(self, radar_token_ids: torch.Tensor,
                         radar_features: Optional[torch.Tensor] = None,
                         max_length: int = 128,
                         num_beams: int = 4) -> list:
        """Generate captions from radar tokens using beam search.

        Args:
            radar_token_ids: (B, M) radar token indices
            radar_features: (B, M, D) optional continuous features
            max_length: maximum generation length
            num_beams: beam search width

        Returns:
            captions: list of str, generated captions
        """
        B = radar_token_ids.shape[0]
        device = radar_token_ids.device

        encoder_input_ids = self.get_radar_input_ids(radar_token_ids)
        encoder_attention_mask = torch.ones_like(encoder_input_ids)

        if radar_features is not None:
            radar_embeds = self.radar_embed_proj(radar_features)
            encoder_outputs = self.t5.encoder(
                inputs_embeds=radar_embeds,
                attention_mask=encoder_attention_mask,
            )
            generated_ids = self.t5.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=encoder_attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )
        else:
            generated_ids = self.t5.generate(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        captions = self.tokenizer.batch_decode(generated_ids,
                                                skip_special_tokens=True)
        return captions
