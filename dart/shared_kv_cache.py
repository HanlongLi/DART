"""Shared Cross-Modal KV Cache for DART.

This module implements the shared key-value cache used by both the
Audio-Query Path (AQP) and the Text-Synthesis Path (TSP).  Each cache
entry stores a (K, V) pair with shape ``(h, d_k)`` together with
per-entry metadata:

* **src** -- origin of the entry (``enc``, ``aqp``, or ``tsp``).
* **pos** -- absolute position index in the sequence.
* **seg** -- segment id (e.g. utterance / chunk boundary).
* **lock** -- ``True`` for committed (finalised) tokens.
* **valid** -- ``True`` while the entry is logically alive.

Conflict resolution
    When TSP writes a token at a position already occupied by an AQP
    entry, the AQP entry's ``valid`` flag is set to ``False`` (the raw
    data is **not** deleted) and the new TSP entry is appended.

Eviction policy
    A sliding-window policy keeps at most *W_tok* AQP/TSP entries and
    *W_enc* encoder entries.  A hard cap of *N_max* is enforced; if
    there is still insufficient room the oldest unlocked entries are
    evicted.
"""

from __future__ import annotations

import torch
from enum import IntEnum
from typing import NamedTuple, Optional


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

class Source(IntEnum):
    """Origin tag for a cache entry."""
    ENC = 0   # Encoder (cross-modal conditioning)
    AQP = 1   # Audio-Query Path (speculative draft)
    TSP = 2   # Text-Synthesis Path (verified output)


class CacheReadOut(NamedTuple):
    """Immutable view returned by :meth:`SharedKVCache.read`.

    Attributes:
        keys:      ``(M, h, d_k)`` selected key vectors.
        values:    ``(M, h, d_k)`` selected value vectors.
        positions: ``(M,)`` absolute position of each entry.
        mask:      ``(M,)`` boolean -- always ``True`` for returned entries
                   (provided so callers can build causal masks downstream).
    """
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor
    mask: torch.Tensor


# ---------------------------------------------------------------------------
# Core cache
# ---------------------------------------------------------------------------

class SharedKVCache:
    """Shared Cross-Modal KV Cache for the DART architecture.

    The cache pre-allocates contiguous storage for *N_max* entries so
    that writes are simple index-scatter operations (GPU-friendly).
    Metadata fields are stored as parallel tensors for efficient
    vectorised filtering.

    Parameters
    ----------
    n_max : int
        Hard upper-bound on the number of entries.
    num_heads : int
        Number of attention heads (*h*).
    d_k : int
        Dimension of each head.
    w_tok : int
        Sliding-window size for AQP / TSP entries.
    w_enc : int
        Sliding-window size for encoder entries.
    device : torch.device, optional
        Target device (default: CPU).
    dtype : torch.dtype, optional
        Floating-point type for K / V tensors (default: float32).
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        n_max: int,
        num_heads: int,
        d_k: int,
        w_tok: int,
        w_enc: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if n_max <= 0:
            raise ValueError(f"n_max must be positive, got {n_max}")
        if w_tok <= 0 or w_enc <= 0:
            raise ValueError("Window sizes w_tok and w_enc must be positive")

        self.n_max = n_max
        self.num_heads = num_heads
        self.d_k = d_k
        self.w_tok = w_tok
        self.w_enc = w_enc
        self.device = device
        self.dtype = dtype

        # ---- pre-allocated K / V storage ----
        self._keys = torch.zeros(
            n_max, num_heads, d_k, device=device, dtype=dtype
        )
        self._values = torch.zeros(
            n_max, num_heads, d_k, device=device, dtype=dtype
        )

        # ---- metadata tensors (parallel arrays) ----
        self._src = torch.full(
            (n_max,), -1, dtype=torch.int8, device=device
        )
        self._pos = torch.full(
            (n_max,), -1, dtype=torch.long, device=device
        )
        self._seg = torch.full(
            (n_max,), -1, dtype=torch.long, device=device
        )
        self._lock = torch.zeros(n_max, dtype=torch.bool, device=device)
        self._valid = torch.zeros(n_max, dtype=torch.bool, device=device)

        # Number of logically valid entries (== self._valid.sum()).
        self._size: int = 0

    # ------------------------------------------------------------ properties
    @property
    def size(self) -> int:
        """Number of logically valid entries currently in the cache."""
        return self._size

    # --------------------------------------------------------- public writes
    def write_enc(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor,
        seg_id: int,
    ) -> torch.Tensor:
        """Write encoder entries into the cache.

        Encoder entries are **always locked** (committed by definition).

        Parameters
        ----------
        keys, values : ``(B, h, d_k)``  -- *B* entries to write.
        positions :    ``(B,)``          -- absolute positions.
        seg_id :       int               -- segment id.

        Returns
        -------
        torch.Tensor
            ``(B,)`` slot indices where the entries were placed.
        """
        return self._write(
            Source.ENC, keys, values, positions, seg_id, lock=True,
        )

    def write_aqp(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor,
        seg_id: int,
        lock: bool = False,
    ) -> torch.Tensor:
        """Write Audio-Query Path entries into the cache.

        Parameters
        ----------
        keys, values : ``(B, h, d_k)``
        positions :    ``(B,)``
        seg_id :       int
        lock :         bool -- set ``True`` to commit the tokens.

        Returns
        -------
        torch.Tensor
            ``(B,)`` slot indices.
        """
        return self._write(
            Source.AQP, keys, values, positions, seg_id, lock=lock,
        )

    def write_tsp(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor,
        seg_id: int,
        lock: bool = False,
    ) -> torch.Tensor:
        """Write Text-Synthesis Path entries into the cache.

        **Conflict resolution** is applied before writing: any valid AQP
        entry whose position matches one of the incoming ``positions``
        will have its ``valid`` flag set to ``False``.

        Parameters
        ----------
        keys, values : ``(B, h, d_k)``
        positions :    ``(B,)``
        seg_id :       int
        lock :         bool

        Returns
        -------
        torch.Tensor
            ``(B,)`` slot indices.
        """
        self._resolve_conflicts(positions)
        return self._write(
            Source.TSP, keys, values, positions, seg_id, lock=lock,
        )

    # ----------------------------------------------------------- public read
    def read(
        self,
        mode: str,
        *,
        suffix_len: Optional[int] = None,
        verify_window: Optional[int] = None,
        current_pos: Optional[int] = None,
    ) -> CacheReadOut:
        """Return a filtered view of the cache.

        Parameters
        ----------
        mode : ``"aqp"`` | ``"tsp"``
            Which path is reading.
        suffix_len : int, optional
            *(AQP mode only)* Keep only the most recent *suffix_len*
            AQP entries (by position).
        verify_window : int, optional
            *(TSP mode only)* Keep only TSP entries within the last
            *verify_window* positions.
        current_pos : int, optional
            Reference position for suffix / window cut-offs.  When
            supplied the filter uses ``pos > current_pos - window``;
            otherwise it simply keeps the top-*k* by position.

        Returns
        -------
        CacheReadOut
            Sorted (by ascending position) selection of K, V, pos, mask.
        """
        if mode == "aqp":
            mask = self._build_aqp_read_mask(suffix_len, current_pos)
        elif mode == "tsp":
            mask = self._build_tsp_read_mask(verify_window, current_pos)
        else:
            raise ValueError(
                f"Unknown read mode: {mode!r}.  Use 'aqp' or 'tsp'."
            )

        # Intersect with global validity
        mask = mask & self._valid

        indices = mask.nonzero(as_tuple=False).reshape(-1)
        if indices.numel() == 0:
            return self._empty_readout()

        # Sort by ascending absolute position (causal ordering)
        sort_order = self._pos[indices].argsort()
        indices = indices[sort_order]

        return CacheReadOut(
            keys=self._keys[indices].clone(),
            values=self._values[indices].clone(),
            positions=self._pos[indices].clone(),
            mask=torch.ones(indices.numel(), dtype=torch.bool, device=self.device),
        )

    # --------------------------------------------------------- public utils
    def lock_entries(self, slots: torch.Tensor) -> None:
        """Mark the entries at *slots* as committed (locked)."""
        self._lock[slots] = True

    def reset(self) -> None:
        """Wipe the entire cache."""
        self._keys.zero_()
        self._values.zero_()
        self._src.fill_(-1)
        self._pos.fill_(-1)
        self._seg.fill_(-1)
        self._lock.fill_(False)
        self._valid.fill_(False)
        self._size = 0

    def stats(self) -> dict[str, int]:
        """Return a snapshot of cache occupancy."""
        v = self._valid
        return {
            "total_slots": self.n_max,
            "used": self._size,
            "free": self.n_max - self._size,
            "enc": int(((self._src == int(Source.ENC)) & v).sum().item()),
            "aqp": int(((self._src == int(Source.AQP)) & v).sum().item()),
            "tsp": int(((self._src == int(Source.TSP)) & v).sum().item()),
            "locked": int((self._lock & v).sum().item()),
        }

    # ------------------------------------------------------------ internals
    def _write(
        self,
        src: Source,
        keys: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor,
        seg_id: int,
        lock: bool,
    ) -> torch.Tensor:
        """Low-level write shared by all three public write methods."""
        count = keys.shape[0]
        if count == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        assert keys.shape == (count, self.num_heads, self.d_k), (
            f"keys shape mismatch: expected ({count}, {self.num_heads}, "
            f"{self.d_k}), got {keys.shape}"
        )
        assert values.shape == keys.shape
        assert positions.shape == (count,)

        slots = self._allocate_slots(count)

        self._keys[slots] = keys
        self._values[slots] = values
        self._src[slots] = int(src)
        self._pos[slots] = positions
        self._seg[slots] = seg_id
        self._lock[slots] = lock
        self._valid[slots] = True

        self._size += count
        return slots

    def _allocate_slots(self, count: int) -> torch.Tensor:
        """Return ``count`` free (invalid) slot indices, evicting if needed."""
        free = self.n_max - self._size
        if free < count:
            self._evict(need=count - free)

        free_mask = ~self._valid
        free_indices = free_mask.nonzero(as_tuple=False).reshape(-1)

        if free_indices.numel() < count:
            raise RuntimeError(
                f"Cannot allocate {count} slots after eviction. "
                f"Cache has {free_indices.numel()} free of {self.n_max} "
                f"({self._size} valid)."
            )
        return free_indices[:count]

    # -------------------------------------------------- conflict resolution
    def _resolve_conflicts(self, tsp_positions: torch.Tensor) -> None:
        """Invalidate every valid AQP entry whose position overlaps
        with the incoming *tsp_positions* tensor.

        The raw K/V data is intentionally **not** zeroed -- only the
        ``valid`` flag is cleared so the slot can be reclaimed later.
        """
        if tsp_positions.numel() == 0:
            return

        aqp_mask = (self._src == int(Source.AQP)) & self._valid
        if not aqp_mask.any():
            return

        aqp_indices = aqp_mask.nonzero(as_tuple=False).reshape(-1)
        aqp_pos = self._pos[aqp_indices]

        # Broadcasting: (|aqp|, 1) == (1, |tsp|) -> (|aqp|, |tsp|)
        overlap = (
            aqp_pos.unsqueeze(1) == tsp_positions.unsqueeze(0)
        ).any(dim=1)

        if overlap.any():
            conflict_slots = aqp_indices[overlap]
            self._valid[conflict_slots] = False
            self._size -= int(conflict_slots.numel())

    # -------------------------------------------------------- read helpers
    def _build_aqp_read_mask(
        self,
        suffix_len: Optional[int],
        current_pos: Optional[int],
    ) -> torch.Tensor:
        """AQP reads **all valid enc** entries + the *suffix_len* most
        recent valid AQP entries (by position)."""
        enc_mask = (self._src == int(Source.ENC)) & self._valid
        aqp_mask = (self._src == int(Source.AQP)) & self._valid

        if suffix_len is not None:
            aqp_mask = self._apply_window(
                aqp_mask, suffix_len, current_pos,
            )

        return enc_mask | aqp_mask

    def _build_tsp_read_mask(
        self,
        verify_window: Optional[int],
        current_pos: Optional[int],
    ) -> torch.Tensor:
        """TSP reads **all valid enc** entries + a *verify_window* of
        recent TSP entries."""
        enc_mask = (self._src == int(Source.ENC)) & self._valid
        tsp_mask = (self._src == int(Source.TSP)) & self._valid

        if verify_window is not None:
            tsp_mask = self._apply_window(
                tsp_mask, verify_window, current_pos,
            )

        return enc_mask | tsp_mask

    def _apply_window(
        self,
        src_mask: torch.Tensor,
        window: int,
        current_pos: Optional[int],
    ) -> torch.Tensor:
        """Restrict *src_mask* to at most *window* most-recent entries.

        If *current_pos* is given, keep entries with
        ``pos > current_pos - window``.  Otherwise fall back to a
        top-*k* selection by position value.
        """
        if current_pos is not None:
            threshold = current_pos - window
            return src_mask & (self._pos > threshold)

        # Fallback: keep top-k by position
        indices = src_mask.nonzero(as_tuple=False).reshape(-1)
        if indices.numel() <= window:
            return src_mask

        positions = self._pos[indices]
        _, topk_local = positions.topk(window)
        keep = indices[topk_local]

        new_mask = torch.zeros_like(src_mask)
        new_mask[keep] = True
        return new_mask

    def _empty_readout(self) -> CacheReadOut:
        """Return an empty ``CacheReadOut`` with correct shapes."""
        return CacheReadOut(
            keys=torch.empty(
                0, self.num_heads, self.d_k,
                device=self.device, dtype=self.dtype,
            ),
            values=torch.empty(
                0, self.num_heads, self.d_k,
                device=self.device, dtype=self.dtype,
            ),
            positions=torch.empty(0, dtype=torch.long, device=self.device),
            mask=torch.empty(0, dtype=torch.bool, device=self.device),
        )

    # -------------------------------------------------------------- eviction
    def _evict(self, need: int) -> None:
        """Free at least *need* slots using the tiered eviction policy.

        Policy tiers (applied in order until enough slots are freed):

        1. **AQP / TSP sliding window** -- for each of AQP and TSP,
           if the number of valid entries exceeds ``w_tok``, evict the
           oldest (by position) excess entries.
        2. **Encoder sliding window** -- same idea with ``w_enc``.
        3. **Oldest-unlocked fallback** -- evict the oldest entries
           that are **not** locked, regardless of source.
        """
        freed = 0

        # ---- Tier 1: AQP / TSP sliding window (w_tok) ----
        for src in (Source.AQP, Source.TSP):
            if freed >= need:
                return
            freed += self._evict_source_window(int(src), self.w_tok)

        # ---- Tier 2: Encoder sliding window (w_enc) ----
        if freed < need:
            freed += self._evict_source_window(int(Source.ENC), self.w_enc)

        # ---- Tier 3: oldest unlocked entries (any source) ----
        if freed < need:
            remaining = need - freed
            unlocked = self._valid & (~self._lock)
            cand = unlocked.nonzero(as_tuple=False).reshape(-1)
            if cand.numel() > 0:
                positions = self._pos[cand]
                _, sorted_idx = positions.sort()
                n_evict = min(remaining, cand.numel())
                evict = cand[sorted_idx[:n_evict]]
                self._valid[evict] = False
                self._size -= int(evict.numel())
                freed += int(evict.numel())

    def _evict_source_window(self, src_int: int, window: int) -> int:
        """Evict excess entries for a single source beyond *window*.

        Returns the number of slots freed.
        """
        src_mask = (self._src == src_int) & self._valid
        indices = src_mask.nonzero(as_tuple=False).reshape(-1)
        excess = indices.numel() - window
        if excess <= 0:
            return 0

        positions = self._pos[indices]
        _, sorted_idx = positions.sort()
        evict = indices[sorted_idx[:excess]]

        self._valid[evict] = False
        self._size -= int(evict.numel())
        return int(evict.numel())

    # -------------------------------------------------------------- repr
    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"SharedKVCache("
            f"N_max={self.n_max}, h={self.num_heads}, d_k={self.d_k}, "
            f"used={s['used']}/{s['total_slots']}, "
            f"enc={s['enc']}, aqp={s['aqp']}, tsp={s['tsp']}, "
            f"locked={s['locked']})"
        )
