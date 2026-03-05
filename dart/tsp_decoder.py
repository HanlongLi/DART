"""TSP Decoder -- deeper Transformer that verifies and corrects AQP drafts.

The Text-Synthesis Path (TSP) operates on a **trailing window** of
the most recent *w* tokens.  It re-attends over encoder context and
its own verified history to produce *corrected* token logits.

Key design choices
------------------
* **Deeper stack** than AQP -- the TSP can afford more layers because
  it runs less frequently (only when a verification cycle is triggered).
* **Conflict resolution** is handled by the shared cache: when the TSP
  writes new K/V entries at positions that already have AQP entries,
  the cache automatically invalidates the stale AQP entries.
* The TSP reads encoder entries plus a bounded *verify_window* of its
  own past entries from the cache.
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
class TSPConfig:
    """Hyper-parameters for the TSP decoder.

    Attributes
    ----------
    vocab_size : int
        Token vocabulary size (must match AQP).
    d_model : int
        Hidden dimension.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of Transformer blocks (deeper than AQP).
    d_ff : int
        Feed-forward intermediate dimension.
    dropout : float
        Dropout rate.
    verify_window : int
        Default trailing-window length (*w*) for verification reads.
    max_len : int
        RoPE table length.
    """
    vocab_size: int = 1024
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    verify_window: int = 16
    max_len: int = 8192


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class TSPOutput:
    """Structured output from :class:`TSPDecoder`.

    Attributes
    ----------
    logits : ``(T, vocab_size)``
        Corrected token classification logits.
    hidden : ``(T, d_model)``
        Final hidden states.
    """
    logits: torch.Tensor
    hidden: torch.Tensor


# ---------------------------------------------------------------------------
# TSP Decoder
# ---------------------------------------------------------------------------

class TSPDecoder(nn.Module):
    """Deeper causal Transformer decoder for the Text-Synthesis Path.

    The TSP consumes a **trailing window** of *w* tokens (draft tokens
    produced by AQP that need verification) and outputs corrected logits.
    All K/V interactions go through the shared cache so that:

    * The TSP can condition on encoder features written by the audio
      front-end.
    * Writing verified tokens back to the cache automatically
      invalidates the speculative AQP entries at the same positions.

    Parameters
    ----------
    cfg : TSPConfig
        All hyper-parameters bundled in a dataclass.
    """

    def __init__(self, cfg: TSPConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # -- Token embedding --
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_drop = nn.Dropout(cfg.dropout)

        # -- Deeper Transformer body --
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

        # -- Single head: corrected logits --
        self.head_logits = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

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
        verify_window: Optional[int] = None,
    ) -> TSPOutput:
        """Produce corrected logits for a trailing window of tokens.

        Parameters
        ----------
        token_ids : ``(T,)``
            Token ids of the trailing window to verify (long).
            Typically ``T == w`` (the verification window width).
        positions : ``(T,)``
            Absolute position indices for each token.
        cache : SharedKVCache
            Shared KV cache instance (same object used by AQP).
        seg_id : int
            Current segment id.
        lock : bool
            Whether to commit these verified tokens.
        verify_window : int | None
            Override the default ``cfg.verify_window`` for this call.
            Controls how many past TSP entries are visible from the
            cache.

        Returns
        -------
        TSPOutput
        """
        vw = verify_window if verify_window is not None else self.cfg.verify_window

        # -- Embed --
        x = self.tok_emb(token_ids)   # (T, d_model)
        x = self.emb_drop(x)

        # -- Transformer blocks --
        for block in self.blocks:
            x = block(
                x, positions, cache, mode="tsp", seg_id=seg_id,
                lock=lock, verify_window=vw,
            )

        x = self.ln_f(x)  # (T, d_model)

        # -- Head --
        logits = self.head_logits(x)  # (T, V)

        return TSPOutput(logits=logits, hidden=x)

    # ---------------------------------------------------------------- utils
    @torch.no_grad()
    def verify_and_correct(
        self,
        draft_ids: torch.Tensor,
        draft_positions: torch.Tensor,
        cache: SharedKVCache,
        seg_id: int,
        *,
        verify_window: Optional[int] = None,
    ) -> TSPOutput:
        """Convenience wrapper for inference-time verification.

        Runs the forward pass in ``torch.no_grad()`` mode with
        ``lock=True`` so that the corrected tokens are committed to the
        cache (and the conflicting AQP entries are invalidated).

        Parameters
        ----------
        draft_ids : ``(T,)``
            Draft token ids produced by AQP.
        draft_positions : ``(T,)``
            Corresponding absolute positions.
        cache, seg_id, verify_window :
            Forwarded to :meth:`forward`.

        Returns
        -------
        TSPOutput
        """
        return self.forward(
            draft_ids, draft_positions, cache, seg_id,
            lock=True, verify_window=verify_window,
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
