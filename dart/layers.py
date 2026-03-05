"""Shared attention primitives for DART decoders.

Both AQP and TSP decoders need a causal multi-head attention layer that
reads its *past* keys/values from a :class:`~dart.shared_kv_cache.SharedKVCache`
rather than from a standard ``past_key_values`` list.

This module provides:

* :class:`CacheAwareMHA` -- Multi-Head Attention that projects the
  current step's Q/K/V, writes the new K/V to the shared cache,
  and reads back the full context (filtered by the cache's read policy).
* :class:`CacheAwareTransformerBlock` -- A pre-norm Transformer block
  (attention -> residual -> FFN -> residual) built on top of
  :class:`CacheAwareMHA`.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dart.shared_kv_cache import SharedKVCache, CacheReadOut


# ---------------------------------------------------------------------------
# Positional helpers
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Pre-computes sin/cos tables up to *max_len* positions and applies
    a fused rotation to (Q, K) pairs.  Supports head dimension *d_k*.
    """

    def __init__(self, d_k: int, max_len: int = 8192, base: float = 10_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_len)

    def _build_cache(self, max_len: int) -> None:
        t = torch.arange(max_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (max_len, d_k/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_len, d_k)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to *q* and *k* at the given absolute *positions*.

        Parameters
        ----------
        q, k : ``(B, h, d_k)`` or ``(M, h, d_k)``
        positions : ``(B,)`` or ``(M,)`` -- absolute position indices.

        Returns
        -------
        (q_rot, k_rot) -- same shapes.
        """
        cos = self.cos_cached[positions]  # (B, d_k)
        sin = self.sin_cached[positions]
        # Expand for heads: (B, 1, d_k)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# Cache-aware Multi-Head Attention
# ---------------------------------------------------------------------------

class CacheAwareMHA(nn.Module):
    """Multi-Head Attention wired to a :class:`SharedKVCache`.

    At each forward step the layer:

    1. Projects the incoming hidden states into Q, K, V.
    2. **Writes** the new K/V into the shared cache (via the
       appropriate ``write_aqp`` / ``write_tsp`` method).
    3. **Reads** back the filtered context from the cache (enc +
       source-specific window).
    4. Applies RoPE to the query and all returned keys, then computes
       scaled dot-product attention with a causal mask derived from
       the position metadata.
    5. Projects the attended values back to ``d_model``.

    Parameters
    ----------
    d_model : int
        Model hidden dimension.
    num_heads : int
        Number of attention heads.
    dropout : float
        Attention dropout probability (applied during training).
    rope_max_len : int
        Maximum sequence length for the RoPE table.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_max_len: int = 8192,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.d_k, max_len=rope_max_len)

    # ------------------------------------------------------------------ fwd
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cache: SharedKVCache,
        mode: str,
        seg_id: int,
        *,
        lock: bool = False,
        suffix_len: Optional[int] = None,
        verify_window: Optional[int] = None,
    ) -> torch.Tensor:
        """Run one attention step through the shared cache.

        Parameters
        ----------
        x : ``(T, d_model)``
            Hidden states for *T* new tokens (no batch dim -- single
            stream for now; batching can be added as a wrapper).
        positions : ``(T,)``
            Absolute position indices for each new token.
        cache : SharedKVCache
            The shared cross-modal KV cache instance.
        mode : ``"aqp"`` | ``"tsp"``
            Which path is calling; determines the cache write/read
            method.
        seg_id : int
            Current segment id.
        lock : bool
            Whether to commit these tokens.
        suffix_len, verify_window : int | None
            Forwarded to ``cache.read()``.

        Returns
        -------
        torch.Tensor ``(T, d_model)``
        """
        T = x.shape[0]

        # -- 1. Project current tokens --
        q = self.W_q(x).view(T, self.num_heads, self.d_k)  # (T, h, d_k)
        k_new = self.W_k(x).view(T, self.num_heads, self.d_k)
        v_new = self.W_v(x).view(T, self.num_heads, self.d_k)

        # -- 2. Write new K/V into the cache --
        write_fn = cache.write_aqp if mode == "aqp" else cache.write_tsp
        write_fn(k_new, v_new, positions, seg_id, lock=lock)

        # -- 3. Read back the full filtered context --
        current_pos = int(positions.max().item()) if T > 0 else None
        readout: CacheReadOut = cache.read(
            mode,
            suffix_len=suffix_len,
            verify_window=verify_window,
            current_pos=current_pos,
        )

        if readout.keys.shape[0] == 0:
            return torch.zeros_like(x)

        k_ctx = readout.keys    # (M, h, d_k)
        v_ctx = readout.values  # (M, h, d_k)
        p_ctx = readout.positions  # (M,)

        # -- 4. Apply RoPE --
        q, _ = self.rope(q, q, positions)       # rotate queries
        k_ctx, _ = self.rope(k_ctx, k_ctx, p_ctx)  # rotate context keys

        # -- 5. Attention scores --
        #   q: (T, h, d_k), k_ctx: (M, h, d_k) -> scores: (h, T, M)
        scale = math.sqrt(self.d_k)
        scores = torch.einsum("thd,mhd->htm", q, k_ctx) / scale  # (h, T, M)

        # Build causal mask: query at pos p can attend to context at pos c
        # only if c <= p
        q_pos = positions.unsqueeze(1)   # (T, 1)
        c_pos = p_ctx.unsqueeze(0)       # (1, M)
        causal = q_pos >= c_pos          # (T, M)
        causal = causal.unsqueeze(0)     # (1, T, M) -> broadcast over heads

        scores = scores.masked_fill(~causal, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        # -- 6. Weighted sum -> output projection --
        #   weights: (h, T, M), v_ctx: (M, h, d_k) -> attn: (T, h, d_k)
        attn = torch.einsum("htm,mhd->thd", weights, v_ctx)
        attn = attn.reshape(T, self.d_model)
        return self.W_o(attn)


# ---------------------------------------------------------------------------
# Pre-norm Transformer block
# ---------------------------------------------------------------------------

class CacheAwareTransformerBlock(nn.Module):
    """Pre-norm Transformer block backed by :class:`CacheAwareMHA`.

    Structure::

        x -+- LayerNorm - CacheAwareMHA - Dropout - (+) -+- LayerNorm - FFN - Dropout - (+) - out
           +-------------------------------------------+   +----------------------------------+

    Parameters
    ----------
    d_model : int
    num_heads : int
    d_ff : int
        Feed-forward intermediate dimension.
    dropout : float
    rope_max_len : int
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        rope_max_len: int = 8192,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CacheAwareMHA(d_model, num_heads, dropout, rope_max_len)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cache: SharedKVCache,
        mode: str,
        seg_id: int,
        *,
        lock: bool = False,
        suffix_len: Optional[int] = None,
        verify_window: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward through attention + FFN with residual connections.

        See :meth:`CacheAwareMHA.forward` for parameter descriptions.
        """
        # Self-attention sub-layer
        h = self.ln1(x)
        h = self.attn(
            h, positions, cache, mode, seg_id,
            lock=lock,
            suffix_len=suffix_len,
            verify_window=verify_window,
        )
        x = x + self.drop1(h)

        # Feed-forward sub-layer
        h = self.ln2(x)
        h = self.ffn(h)
        x = x + self.drop2(h)
        return x
