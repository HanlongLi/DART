"""AQP Decoder -- lightweight causal Transformer with dual output heads.

The Audio-Query Path (AQP) is the *speculative draft* decoder in the
DART architecture.  It is deliberately shallow so that it can run with
minimal latency in a streaming setting.

Outputs
-------
* **logits** -- token-level classification logits over the vocabulary.
* **prosody_plan** -- per-token prosody features:
    - duration          (scalar, regressed or binned)
    - coarse F0 bin     (discrete, softmax over ``n_f0_bins``)
    - coarse log-energy bin (discrete, softmax over ``n_energy_bins``)

All attention layers use :class:`~dart.layers.CacheAwareTransformerBlock`
wired to a shared :class:`~dart.shared_kv_cache.SharedKVCache`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from dart.shared_kv_cache import SharedKVCache
from dart.layers import CacheAwareTransformerBlock


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AQPConfig:
    """Hyper-parameters for the AQP decoder.

    Attributes
    ----------
    vocab_size : int
        Token vocabulary size.
    d_model : int
        Hidden dimension.
    num_heads : int
        Number of attention heads (must divide ``d_model``).
    num_layers : int
        Number of Transformer blocks (kept small for low latency).
    d_ff : int
        Feed-forward intermediate dimension.
    dropout : float
        Dropout rate.
    n_f0_bins : int
        Number of coarse fundamental-frequency bins.
    n_energy_bins : int
        Number of coarse log-energy bins.
    max_len : int
        RoPE table length.
    """
    vocab_size: int = 1024
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    n_f0_bins: int = 64
    n_energy_bins: int = 32
    max_len: int = 8192


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class AQPOutput:
    """Structured output from :class:`AQPDecoder`.

    Attributes
    ----------
    logits : ``(T, vocab_size)``
        Token classification logits.
    duration : ``(T, 1)``
        Predicted duration (scalar per token).
    f0_logits : ``(T, n_f0_bins)``
        Coarse F0 bin logits.
    energy_logits : ``(T, n_energy_bins)``
        Coarse log-energy bin logits.
    hidden : ``(T, d_model)``
        Final hidden states (useful for downstream probing / losses).
    """
    logits: torch.Tensor
    duration: torch.Tensor
    f0_logits: torch.Tensor
    energy_logits: torch.Tensor
    hidden: torch.Tensor


# ---------------------------------------------------------------------------
# AQP Decoder
# ---------------------------------------------------------------------------

class AQPDecoder(nn.Module):
    """Lightweight causal Transformer decoder for the Audio-Query Path.

    This is deliberately *shallow* (``num_layers`` is typically 2--4)
    so that speculative draft tokens can be produced at low latency.

    Parameters
    ----------
    cfg : AQPConfig
        All hyper-parameters bundled in a dataclass.
    """

    def __init__(self, cfg: AQPConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # -- Token embedding --
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_drop = nn.Dropout(cfg.dropout)

        # -- Transformer body --
        self.blocks = nn.ModuleList([
            CacheAwareTransformerBlock(
                d_model=cfg.d_model,
                num_heads=cfg.num_heads,
                d_ff=cfg.d_ff,
                dropout=cfg.dropout,
                rope_max_len=cfg.max_len,
            )
            for _ in range(cfg.num_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # -- Head 1: token logits --
        self.head_logits = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # -- Head 2: prosody plan --
        #   Shared prosody trunk -> three sub-heads
        self.prosody_trunk = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
        )
        self.head_duration = nn.Linear(cfg.d_model, 1)
        self.head_f0 = nn.Linear(cfg.d_model, cfg.n_f0_bins)
        self.head_energy = nn.Linear(cfg.d_model, cfg.n_energy_bins)

        self._init_weights()

    # ------------------------------------------------------------------ fwd
    def forward(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        cache: SharedKVCache,
        seg_id: int,
        *,
        lock: bool = False,
        suffix_len: Optional[int] = None,
    ) -> AQPOutput:
        """Produce draft logits and a prosody plan.

        Parameters
        ----------
        token_ids : ``(T,)``
            Input token ids (long).
        positions : ``(T,)``
            Absolute position indices.
        cache : SharedKVCache
            Shared KV cache instance.
        seg_id : int
            Current segment id.
        lock : bool
            Commit these entries in the cache.
        suffix_len : int | None
            Controls how many past AQP entries are visible.

        Returns
        -------
        AQPOutput
        """
        T = token_ids.shape[0]

        # -- Embed --
        x = self.tok_emb(token_ids)   # (T, d_model)
        x = self.emb_drop(x)

        # -- Transformer blocks --
        for block in self.blocks:
            x = block(
                x, positions, cache, mode="aqp", seg_id=seg_id,
                lock=lock, suffix_len=suffix_len,
            )

        x = self.ln_f(x)  # (T, d_model)

        # -- Heads --
        logits = self.head_logits(x)          # (T, V)

        pros = self.prosody_trunk(x)          # (T, d_model)
        duration = self.head_duration(pros)   # (T, 1)
        f0 = self.head_f0(pros)               # (T, n_f0_bins)
        energy = self.head_energy(pros)       # (T, n_energy_bins)

        return AQPOutput(
            logits=logits,
            duration=duration,
            f0_logits=f0,
            energy_logits=energy,
            hidden=x,
        )

    # -------------------------------------------------------------- init
    def _init_weights(self) -> None:
        """Xavier-uniform for linear layers, normal for embeddings."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
