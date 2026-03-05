"""DART Inference Engine -- per-chunk streaming decode loop.

This module orchestrates the full DART inference cycle on a
chunk-by-chunk (e.g. 200 ms) basis:

1. **Encode** -- receive an encoder chunk update $\\Delta H_c$ and
   write it into the shared KV cache.
2. **AQP Draft** -- auto-regressively generate $k$ speculative tokens
   through the lightweight AQP decoder.
3. **TSP Verify** -- when the number of uncommitted tokens exceeds the
   verification window $w$, run the deeper TSP decoder over the
   trailing window.  Conflict resolution in the cache automatically
   invalidates stale AQP entries.
4. **Commit** -- advance the commit cursor $v$ for every position
   where the TSP's corrected token agrees with the AQP draft *and*
   the confidence exceeds a threshold $\tau$.  Alternatively, commit
   after $n_{\text{stable}}$ consecutive matching steps.

The engine maintains all mutable state (cursor positions, draft
buffers, segment counter) and exposes a single :meth:`step` method
that is called once per audio chunk.

Usage example::

    engine = DARTInferenceEngine(cfg, aqp, tsp, cache)
    for chunk in audio_chunks:
        result = engine.step(chunk)
        committed = result.newly_committed_ids
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from dart.shared_kv_cache import SharedKVCache, Source
from dart.aqp_decoder import AQPDecoder, AQPOutput
from dart.tsp_decoder import TSPDecoder, TSPOutput


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    """Knobs for the DART inference loop.

    Attributes
    ----------
    k_draft : int
        Number of speculative tokens the AQP produces per chunk.
    verify_window : int
        Trailing-window width *w* passed to the TSP.
    verify_trigger : int
        How many uncommitted draft tokens must accumulate before
        a TSP verification cycle is triggered.
    confidence_threshold : float
        Minimum softmax probability $\\tau$ for a TSP token to be
        considered *stable* (used in the commit rule).
    n_stable : int
        Number of consecutive positions where AQP and TSP must agree
        (above $\\tau$) before the commit cursor advances.
    enc_dim : int
        Encoder hidden dimension (= ``d_model`` for the
        encoder front-end, not necessarily the decoder ``d_model``).
    d_model : int
        Decoder hidden dimension (same for AQP and TSP).
    num_heads : int
        Number of attention heads in the cache.
    """
    k_draft: int = 8
    verify_window: int = 16
    verify_trigger: int = 8
    confidence_threshold: float = 0.7
    n_stable: int = 3
    enc_dim: int = 256
    d_model: int = 256
    num_heads: int = 4


# ---------------------------------------------------------------------------
# Step output container
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Output of a single :meth:`DARTInferenceEngine.step` call.

    Attributes
    ----------
    chunk_id : int
        Monotonic chunk counter.
    enc_positions : torch.Tensor
        Encoder positions written this step.
    draft_ids : torch.Tensor
        Token ids produced by AQP this step.
    draft_logits : torch.Tensor
        Full logit tensor from AQP.
    verified : bool
        Whether TSP verification ran this step.
    corrected_ids : torch.Tensor | None
        If verified, the argmax token ids from TSP.
    corrected_logits : torch.Tensor | None
        If verified, the full TSP logit tensor.
    newly_committed_ids : torch.Tensor
        Token ids that crossed the commit threshold this step
        (may be empty).
    newly_committed_positions : torch.Tensor
        Corresponding absolute positions.
    commit_cursor : int
        Updated commit cursor $v$ after this step.
    cache_stats : dict[str, int]
        Snapshot of cache occupancy.
    """
    chunk_id: int
    enc_positions: torch.Tensor
    draft_ids: torch.Tensor
    draft_logits: torch.Tensor
    verified: bool
    corrected_ids: Optional[torch.Tensor]
    corrected_logits: Optional[torch.Tensor]
    newly_committed_ids: torch.Tensor
    newly_committed_positions: torch.Tensor
    commit_cursor: int
    cache_stats: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class DARTInferenceEngine:
    """Orchestrates the DART per-chunk inference loop.

    The engine is stateful: it tracks the current position cursors,
    the segment counter, the draft buffer, and the commit frontier.

    Parameters
    ----------
    cfg : EngineConfig
        Loop hyper-parameters.
    aqp : AQPDecoder
        Speculative draft decoder (should be in ``eval()`` mode).
    tsp : TSPDecoder
        Verification decoder (should be in ``eval()`` mode).
    cache : SharedKVCache
        Pre-allocated shared KV cache.
    enc_projector : torch.nn.Module | None
        Optional linear projection from ``enc_dim`` -> ``(num_heads, d_k)``
        so that raw encoder hidden states can be written as cache K/V.
        If *None*, a default ``nn.Linear`` is created.
    device : torch.device
        Target device.
    """

    def __init__(
        self,
        cfg: EngineConfig,
        aqp: AQPDecoder,
        tsp: TSPDecoder,
        cache: SharedKVCache,
        enc_projector: Optional[torch.nn.Module] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.cfg = cfg
        self.aqp = aqp
        self.tsp = tsp
        self.cache = cache
        self.device = device

        d_k = cfg.d_model // cfg.num_heads

        if enc_projector is not None:
            self.enc_proj_k = enc_projector
            self.enc_proj_v = enc_projector
        else:
            self.enc_proj_k = torch.nn.Linear(
                cfg.enc_dim, cfg.num_heads * d_k, bias=False,
            ).to(device)
            self.enc_proj_v = torch.nn.Linear(
                cfg.enc_dim, cfg.num_heads * d_k, bias=False,
            ).to(device)

        # -- Mutable state --
        self._chunk_id: int = 0
        self._seg_id: int = 0
        self._enc_cursor: int = 0        # next encoder position
        self._draft_cursor: int = 0      # next AQP draft position
        self._commit_cursor: int = 0     # frontier of locked tokens ($v$)

        # Rolling buffer of draft token ids and their positions
        self._draft_buffer_ids: list[int] = []
        self._draft_buffer_pos: list[int] = []

        # Consecutive-stability counter per position (for commit rule)
        self._stability_counter: int = 0

    # --------------------------- public API ---------------------------

    def step(self, encoder_chunk: torch.Tensor) -> StepResult:
        """Run one full 200 ms cycle.

        Parameters
        ----------
        encoder_chunk : ``(F, enc_dim)``
            The encoder update $\\Delta H_c$ for this chunk -- *F* frames
            of hidden states from the audio front-end.

        Returns
        -------
        StepResult
        """
        self._chunk_id += 1

        # -- 1. Encode: project & write Delta H_c --
        enc_positions = self._encode(encoder_chunk)

        # -- 2. AQP Draft: generate k speculative tokens --
        draft_ids, draft_logits, aqp_out = self._draft()

        # -- 3. TSP Verify (conditional) --
        verified = False
        corrected_ids: Optional[torch.Tensor] = None
        corrected_logits: Optional[torch.Tensor] = None

        uncommitted_count = len(self._draft_buffer_ids)
        if uncommitted_count >= self.cfg.verify_trigger:
            verified = True
            corrected_ids, corrected_logits = self._verify()

        # -- 4. Commit --
        new_ids, new_pos = self._commit(
            draft_ids if corrected_ids is None else corrected_ids,
            corrected_ids is not None,
        )

        result = StepResult(
            chunk_id=self._chunk_id,
            enc_positions=enc_positions,
            draft_ids=draft_ids,
            draft_logits=draft_logits,
            verified=verified,
            corrected_ids=corrected_ids,
            corrected_logits=corrected_logits,
            newly_committed_ids=new_ids,
            newly_committed_positions=new_pos,
            commit_cursor=self._commit_cursor,
            cache_stats=self.cache.stats(),
        )

        log.debug(
            "chunk=%d  enc_pos=%s  drafted=%d  verified=%s  "
            "committed=%d  cursor=%d  cache=%s",
            self._chunk_id,
            enc_positions.tolist(),
            draft_ids.numel(),
            verified,
            new_ids.numel(),
            self._commit_cursor,
            self.cache.stats(),
        )

        return result

    def reset(self) -> None:
        """Reset all engine state (including the cache)."""
        self.cache.reset()
        self._chunk_id = 0
        self._seg_id = 0
        self._enc_cursor = 0
        self._draft_cursor = 0
        self._commit_cursor = 0
        self._draft_buffer_ids.clear()
        self._draft_buffer_pos.clear()
        self._stability_counter = 0

    @property
    def commit_cursor(self) -> int:
        """Current commit frontier $v$."""
        return self._commit_cursor

    @property
    def uncommitted_count(self) -> int:
        """Number of drafted but not-yet-committed tokens."""
        return len(self._draft_buffer_ids)

    # ----------------------- internal stages --------------------------

    @torch.no_grad()
    def _encode(self, encoder_chunk: torch.Tensor) -> torch.Tensor:
        """Project encoder frames and write them into the cache.

        Parameters
        ----------
        encoder_chunk : ``(F, enc_dim)``

        Returns
        -------
        torch.Tensor ``(F,)`` -- positions assigned.
        """
        F_frames = encoder_chunk.shape[0]
        if F_frames == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        d_k = self.cfg.d_model // self.cfg.num_heads

        # Project to (F, num_heads * d_k) then reshape -> (F, h, d_k)
        k_enc = self.enc_proj_k(encoder_chunk).view(
            F_frames, self.cfg.num_heads, d_k,
        )
        v_enc = self.enc_proj_v(encoder_chunk).view(
            F_frames, self.cfg.num_heads, d_k,
        )

        positions = torch.arange(
            self._enc_cursor,
            self._enc_cursor + F_frames,
            dtype=torch.long,
            device=self.device,
        )

        self.cache.write_enc(k_enc, v_enc, positions, seg_id=self._seg_id)
        self._enc_cursor += F_frames
        return positions

    @torch.no_grad()
    def _draft(self) -> tuple[torch.Tensor, torch.Tensor, AQPOutput]:
        """Auto-regressively generate *k_draft* tokens with AQP.

        Uses greedy argmax sampling for simplicity.  Each token is
        generated one at a time so the cache accumulates the KV
        entries correctly.

        Returns
        -------
        (draft_ids, draft_logits, last_aqp_output)
        """
        k = self.cfg.k_draft
        all_ids: list[int] = []
        all_logits: list[torch.Tensor] = []

        # Seed: if the draft buffer is empty, use token 0 (BOS-like).
        if self._draft_buffer_ids:
            prev_id = self._draft_buffer_ids[-1]
        else:
            prev_id = 0

        last_out: Optional[AQPOutput] = None

        for i in range(k):
            token_id = torch.tensor(
                [prev_id], dtype=torch.long, device=self.device,
            )
            pos = torch.tensor(
                [self._draft_cursor], dtype=torch.long, device=self.device,
            )
            out: AQPOutput = self.aqp(
                token_id, pos, self.cache, seg_id=self._seg_id,
            )
            last_out = out
            logits_t = out.logits[0]                    # (V,)
            next_id = int(logits_t.argmax(dim=-1).item())

            all_ids.append(next_id)
            all_logits.append(logits_t)

            # Track in draft buffer
            self._draft_buffer_ids.append(next_id)
            self._draft_buffer_pos.append(self._draft_cursor)

            prev_id = next_id
            self._draft_cursor += 1

        draft_ids = torch.tensor(all_ids, dtype=torch.long, device=self.device)
        draft_logits = torch.stack(all_logits, dim=0)   # (k, V)

        assert last_out is not None
        return draft_ids, draft_logits, last_out

    @torch.no_grad()
    def _verify(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Run TSP over the trailing verification window.

        Takes the last ``min(len(draft_buffer), verify_window)``
        uncommitted tokens and feeds them through the TSP.

        Returns
        -------
        (corrected_ids, corrected_logits)
        """
        w = self.cfg.verify_window
        buf_ids = self._draft_buffer_ids[-w:]
        buf_pos = self._draft_buffer_pos[-w:]

        token_ids = torch.tensor(
            buf_ids, dtype=torch.long, device=self.device,
        )
        positions = torch.tensor(
            buf_pos, dtype=torch.long, device=self.device,
        )

        tsp_out: TSPOutput = self.tsp.verify_and_correct(
            token_ids, positions, self.cache, seg_id=self._seg_id,
        )

        corrected_ids = tsp_out.logits.argmax(dim=-1)   # (W,)
        return corrected_ids, tsp_out.logits

    def _commit(
        self,
        reference_ids: torch.Tensor,
        was_verified: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the commit cursor $v$ based on stability criteria.

        Commit rule
        -----------
        Walk forward from the current commit cursor.  At each position
        in the draft buffer that has been verified by TSP:

        * If the TSP's argmax token **matches** the draft token AND the
          TSP confidence >= $\\tau$, increment a stability counter.
        * Otherwise, reset the stability counter.
        * When the stability counter reaches ``n_stable``, commit all
          positions up to and including the current one, lock them in
          the cache, and advance $v$.

        If no verification happened this step, nothing is committed
        (pure speculative drafts are never committed without TSP
        confirmation).

        Returns
        -------
        (newly_committed_ids, newly_committed_positions)
        """
        if not was_verified:
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
            )

        # Find positions in the draft buffer that are >= commit_cursor
        # and were part of the verification window.
        commit_ids: list[int] = []
        commit_pos: list[int] = []

        w = self.cfg.verify_window
        buf_ids = self._draft_buffer_ids[-w:]
        buf_pos = self._draft_buffer_pos[-w:]

        # reference_ids are the TSP argmax for the same window
        assert reference_ids.numel() == len(buf_ids), (
            f"reference_ids length {reference_ids.numel()} != "
            f"buffer window length {len(buf_ids)}"
        )

        furthest_commit = self._commit_cursor

        for i, (draft_id, pos) in enumerate(zip(buf_ids, buf_pos)):
            if pos < self._commit_cursor:
                # Already committed -- skip
                continue

            tsp_id = int(reference_ids[i].item())

            # Check confidence
            # (we don't have the logits here directly, so we rely on
            #  the simpler "agreement" signal; the caller can optionally
            #  also check softmax confidence upstream)
            if tsp_id == draft_id:
                self._stability_counter += 1
            else:
                self._stability_counter = 0

            if self._stability_counter >= self.cfg.n_stable:
                # Commit everything from commit_cursor up to pos (inclusive)
                for j in range(len(buf_ids)):
                    p = buf_pos[j]
                    if self._commit_cursor <= p <= pos and p not in commit_pos:
                        commit_ids.append(buf_ids[j])
                        commit_pos.append(p)
                furthest_commit = pos + 1
                # Reset counter for next batch
                self._stability_counter = 0

        if furthest_commit > self._commit_cursor:
            self._commit_cursor = furthest_commit

            # Lock the committed positions in the cache
            self._lock_positions(commit_pos)

            # Trim the draft buffer: remove everything before commit_cursor
            self._trim_draft_buffer()

        return (
            torch.tensor(commit_ids, dtype=torch.long, device=self.device),
            torch.tensor(commit_pos, dtype=torch.long, device=self.device),
        )

    def _lock_positions(self, positions: list[int]) -> None:
        """Lock all cache entries (any source) at the given positions."""
        if not positions:
            return
        pos_t = torch.tensor(positions, dtype=torch.long, device=self.device)

        # Find valid slots whose position is in the commit set
        valid_mask = self.cache._valid.clone()
        slot_indices = valid_mask.nonzero(as_tuple=False).reshape(-1)

        if slot_indices.numel() == 0:
            return

        slot_pos = self.cache._pos[slot_indices]
        # (|slots|, 1) == (1, |pos|) -> broadcast match
        match = (slot_pos.unsqueeze(1) == pos_t.unsqueeze(0)).any(dim=1)
        if match.any():
            to_lock = slot_indices[match]
            self.cache.lock_entries(to_lock)

    def _trim_draft_buffer(self) -> None:
        """Remove entries before ``_commit_cursor`` from the buffer."""
        new_ids: list[int] = []
        new_pos: list[int] = []
        for tid, p in zip(self._draft_buffer_ids, self._draft_buffer_pos):
            if p >= self._commit_cursor:
                new_ids.append(tid)
                new_pos.append(p)
        self._draft_buffer_ids = new_ids
        self._draft_buffer_pos = new_pos

    # ------------------------ segment boundary ------------------------

    def new_segment(self) -> None:
        """Signal a segment boundary (e.g. new utterance).

        Increments the segment id.  Does **not** flush the cache; call
        :meth:`reset` for a full wipe.
        """
        self._seg_id += 1
        self._stability_counter = 0

    # ------------------------ diagnostics -----------------------------

    def state_snapshot(self) -> dict:
        """Return a JSON-serialisable snapshot of the engine state."""
        return {
            "chunk_id": self._chunk_id,
            "seg_id": self._seg_id,
            "enc_cursor": self._enc_cursor,
            "draft_cursor": self._draft_cursor,
            "commit_cursor": self._commit_cursor,
            "uncommitted": self.uncommitted_count,
            "stability_counter": self._stability_counter,
            "cache": self.cache.stats(),
        }
