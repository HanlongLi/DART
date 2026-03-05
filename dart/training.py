"""Staged training curriculum for the DART architecture.

This module implements the three-stage training pipeline described in
the DART architecture:

Stage A -- Warm starts
    Separate pre-training of the Encoder (CTC), AQP (draft + prosody
    supervision), and TSP (distillation from a teacher model).

Stage B -- Cross-path alignment
    End-to-end alignment of AQP and TSP through the shared KV cache
    using the contrastive alignment loss.  Includes "KV-drop" on
    non-committed cache entries with a dropout probability that ramps
    linearly from 0 to ``kv_drop_max`` over the stage.

Stage C -- Joint end-to-end optimisation
    All components trained jointly under the full latency-aware loss
    with AdamW, mixed-precision (``torch.cuda.amp``), gradient clipping,
    and cosine learning-rate decay.

Each ``train_stage_*`` function accepts pre-built models, data loaders,
and configuration dataclasses.  They are designed for multi-GPU servers
using ``torch.nn.parallel.DistributedDataParallel`` (DDP) but also run
correctly on a single device.

Usage example (single-node, multi-GPU)::

    torchrun --nproc_per_node=4 run_training.py --stage a
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from dart.shared_kv_cache import SharedKVCache, Source
from dart.aqp_decoder import AQPDecoder, AQPConfig, AQPOutput
from dart.tsp_decoder import TSPDecoder, TSPConfig, TSPOutput
from dart.losses import (
    ContrastiveAlignmentLoss,
    ProsodyLoss,
    DARTJointLoss,
    LossWeights,
    LossBreakdown,
    LatencyLoss,
)
from dart.data import DARTBatch

log = logging.getLogger(__name__)


# =====================================================================
# Configuration dataclasses
# =====================================================================

@dataclass
class StageAConfig:
    """Hyper-parameters for Stage A (warm starts).

    Attributes
    ----------
    epochs : int
        Number of training epochs for each sub-stage.
    lr : float
        Peak learning rate.
    weight_decay : float
        AdamW weight decay.
    grad_clip : float
        Maximum gradient norm.
    ctc_blank_id : int
        Blank token id for the CTC loss.
    teacher_temperature : float
        Temperature for TSP distillation from the teacher.
    use_amp : bool
        Enable automatic mixed precision.
    log_every : int
        Log metrics every *n* steps.
    """
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    ctc_blank_id: int = 0
    teacher_temperature: float = 2.0
    use_amp: bool = True
    log_every: int = 50


@dataclass
class StageBConfig:
    """Hyper-parameters for Stage B (cross-path alignment).

    Attributes
    ----------
    epochs : int
        Number of alignment training epochs.
    lr : float
        Peak learning rate.
    weight_decay : float
        AdamW weight decay.
    grad_clip : float
        Maximum gradient norm.
    kv_drop_max : float
        Terminal KV-drop probability (ramped linearly from 0).
    gamma : float
        Speaker-consistency penalty weight in the alignment loss.
    d_align : int
        Embedding dimension for contrastive alignment.
    d_speaker : int
        Speaker embedding dimension.
    temperature : float
        InfoNCE temperature.
    use_amp : bool
        Enable automatic mixed precision.
    log_every : int
        Log metrics every *n* steps.
    """
    epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    kv_drop_max: float = 0.2
    gamma: float = 1.0
    d_align: int = 128
    d_speaker: int = 64
    temperature: float = 0.07
    use_amp: bool = True
    log_every: int = 50


@dataclass
class StageCConfig:
    """Hyper-parameters for Stage C (joint end-to-end optimisation).

    Attributes
    ----------
    epochs : int
        Number of training epochs.
    lr : float
        Peak learning rate.
    weight_decay : float
        AdamW weight decay.
    grad_clip : float
        Maximum gradient norm.
    warmup_steps : int
        Linear warmup steps before cosine decay begins.
    use_amp : bool
        Enable automatic mixed precision.
    loss_weights : LossWeights
        Per-term multipliers for the joint loss.
    log_every : int
        Log metrics every *n* steps.
    """
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    use_amp: bool = True
    loss_weights: LossWeights = field(default_factory=LossWeights)
    log_every: int = 50


# =====================================================================
# Shared helpers
# =====================================================================

def _rank() -> int:
    """Return the DDP rank, or 0 if not in a distributed context."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def _is_main() -> bool:
    """Return ``True`` on rank-0 (or when not running distributed)."""
    return _rank() == 0


def _log_metrics(
    stage: str,
    epoch: int,
    step: int,
    metrics: dict[str, float],
) -> None:
    """Emit a structured log line on the main process only."""
    if not _is_main():
        return
    parts = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    log.info("[%s] epoch=%d step=%d | %s", stage, epoch, step, parts)


def _build_optimizer(
    params,
    lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Construct an AdamW optimiser with standard betas."""
    return torch.optim.AdamW(
        params, lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay,
    )


def _cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay to zero.

    Parameters
    ----------
    optimizer : Optimizer
    warmup_steps : int
        Steps of linear warm-up.
    total_steps : int
        Total training steps (warmup + decay).
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(
            1, total_steps - warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =====================================================================
# KV-drop helper
# =====================================================================

def kv_drop(
    cache: SharedKVCache,
    drop_prob: float,
    generator: Optional[torch.Generator] = None,
) -> None:
    """Stochastically mask non-committed cache entries.

    For each valid, unlocked entry the ``valid`` flag is set to
    ``False`` with probability *drop_prob*.  Locked (committed) entries
    are never dropped.  The cache's internal size counter is updated
    accordingly.

    This implements the "KV-drop" regularisation mechanism described
    in Stage B of the DART curriculum.

    Parameters
    ----------
    cache : SharedKVCache
        The shared KV cache to apply dropout on.
    drop_prob : float
        Probability of dropping each non-committed entry.
    generator : torch.Generator, optional
        RNG for reproducible masking.
    """
    if drop_prob <= 0.0:
        return

    eligible = cache._valid & (~cache._lock)
    if not eligible.any():
        return

    indices = eligible.nonzero(as_tuple=False).reshape(-1)
    mask = torch.rand(
        indices.numel(), device=cache.device, generator=generator,
    ) < drop_prob

    if mask.any():
        drop_slots = indices[mask]
        cache._valid[drop_slots] = False
        cache._size -= int(drop_slots.numel())


# =====================================================================
# Encoder stub (used in Stage A)
# =====================================================================

class EncoderWrapper(nn.Module):
    """Thin wrapper that pairs an arbitrary encoder with a CTC head.

    The encoder is expected to map raw features to hidden states.
    This wrapper adds a linear projection to the vocabulary for CTC
    training.

    Parameters
    ----------
    encoder : nn.Module
        Audio encoder mapping ``(B, F, feat_dim)`` to ``(B, F, enc_dim)``.
    enc_dim : int
        Output dimension of the encoder.
    vocab_size : int
        Token vocabulary size (including the CTC blank).
    """

    def __init__(
        self,
        encoder: nn.Module,
        enc_dim: int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.ctc_proj = nn.Linear(enc_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return CTC logits ``(B, F, vocab_size)``."""
        h = self.encoder(x)     # (B, F, enc_dim)
        return self.ctc_proj(h)  # (B, F, vocab_size)


# =====================================================================
# Stage A -- Warm Starts
# =====================================================================

def train_stage_a(
    *,
    encoder: nn.Module,
    aqp: AQPDecoder,
    tsp: TSPDecoder,
    teacher: nn.Module,
    train_loader: DataLoader,
    cache_factory,
    cfg: StageAConfig,
    device: torch.device,
) -> dict[str, list[float]]:
    """Stage A: separate pre-training of encoder, AQP, and TSP.

    Three sequential sub-stages run within a single call:

    A-1  **Encoder warm-up** (CTC loss)
         The encoder is trained with ``torch.nn.CTCLoss`` to align
         audio frames with target token sequences.

    A-2  **AQP warm-up** (draft cross-entropy + prosody supervision)
         The lightweight decoder is trained with teacher-forced token
         predictions and prosody targets (duration MSE, F0 CE, energy
         CE).

    A-3  **TSP distillation** (KL divergence from teacher)
         The deeper decoder is trained to match a pre-trained teacher
         model's output distribution using KL divergence at a
         configurable temperature.

    Parameters
    ----------
    encoder : nn.Module
        Audio encoder wrapped in :class:`EncoderWrapper` (provides a
        ``ctc_proj`` head).
    aqp : AQPDecoder
        AQP decoder to warm-start.
    tsp : TSPDecoder
        TSP decoder to warm-start via distillation.
    teacher : nn.Module
        Pre-trained teacher whose logits are the distillation target.
        Must accept the same inputs as ``tsp`` and return an object
        with a ``.logits`` attribute.
    train_loader : DataLoader
        Yields :class:`~dart.data.DARTBatch` instances.
    cache_factory : callable
        ``cache_factory()`` returns a fresh :class:`SharedKVCache`.
        A new cache is created per sample to avoid cross-contamination.
    cfg : StageAConfig
    device : torch.device

    Returns
    -------
    dict[str, list[float]]
        Per-step loss history keyed by ``"enc_ctc"``, ``"aqp_draft"``,
        ``"aqp_pros"``, ``"tsp_distill"``.
    """
    history: dict[str, list[float]] = {
        "enc_ctc": [], "aqp_draft": [], "aqp_pros": [], "tsp_distill": [],
    }

    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)
    prosody_loss_fn = ProsodyLoss()

    # -- A-1: Encoder CTC --
    log.info("Stage A-1: Encoder warm-up (CTC)")
    encoder.train()
    enc_opt = _build_optimizer(encoder.parameters(), cfg.lr, cfg.weight_decay)
    ctc_loss_fn = nn.CTCLoss(blank=cfg.ctc_blank_id, reduction="mean")

    for epoch in range(cfg.epochs):
        for step, batch in enumerate(train_loader):
            batch: DARTBatch = batch.to(device)
            enc_opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                # Encoder forward: (B, F_max, enc_dim) -> CTC logits
                ctc_logits = encoder(batch.audio_features)  # (B, F, V)
                log_probs = F.log_softmax(ctc_logits, dim=-1)
                # CTCLoss expects (T, B, C) ordering
                log_probs_t = log_probs.permute(1, 0, 2)

                loss = ctc_loss_fn(
                    log_probs_t,
                    batch.token_ids,
                    batch.audio_lens,
                    batch.token_lens,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(enc_opt)
            nn.utils.clip_grad_norm_(encoder.parameters(), cfg.grad_clip)
            scaler.step(enc_opt)
            scaler.update()

            history["enc_ctc"].append(loss.item())
            if step % cfg.log_every == 0:
                _log_metrics("A-1", epoch, step, {"ctc": loss.item()})

    # -- A-2: AQP (draft + prosody) --
    log.info("Stage A-2: AQP warm-up (draft + prosody)")
    aqp.train()
    aqp_opt = _build_optimizer(aqp.parameters(), cfg.lr, cfg.weight_decay)

    for epoch in range(cfg.epochs):
        for step, batch in enumerate(train_loader):
            batch: DARTBatch = batch.to(device)
            aqp_opt.zero_grad(set_to_none=True)

            cache = cache_factory()

            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                B, T = batch.token_ids.shape

                draft_loss_accum = torch.tensor(0.0, device=device)
                pros_loss_accum = torch.tensor(0.0, device=device)

                for b in range(B):
                    length = int(batch.token_lens[b].item())
                    if length < 2:
                        continue

                    ids_b = batch.token_ids[b, :length]
                    positions = torch.arange(
                        length, dtype=torch.long, device=device,
                    )

                    # Teacher-forced: feed tokens 0..T-2, predict 1..T-1
                    out: AQPOutput = aqp(
                        ids_b[:-1], positions[:-1], cache, seg_id=0,
                    )

                    # Draft cross-entropy
                    draft_loss = F.cross_entropy(
                        out.logits, ids_b[1:],
                    )
                    draft_loss_accum = draft_loss_accum + draft_loss

                    # Prosody supervision
                    T_out = out.logits.shape[0]
                    pros = prosody_loss_fn(
                        out.duration[:T_out],
                        batch.durations[b, 1:length],
                        out.f0_logits[:T_out],
                        batch.f0_bins[b, 1:length],
                        out.energy_logits[:T_out],
                        batch.energy_bins[b, 1:length],
                    )
                    pros_loss_accum = pros_loss_accum + pros

                    cache.reset()

                total_loss = (draft_loss_accum + pros_loss_accum) / max(B, 1)

            scaler.scale(total_loss).backward()
            scaler.unscale_(aqp_opt)
            nn.utils.clip_grad_norm_(aqp.parameters(), cfg.grad_clip)
            scaler.step(aqp_opt)
            scaler.update()

            history["aqp_draft"].append(draft_loss_accum.item() / max(B, 1))
            history["aqp_pros"].append(pros_loss_accum.item() / max(B, 1))
            if step % cfg.log_every == 0:
                _log_metrics("A-2", epoch, step, {
                    "draft": history["aqp_draft"][-1],
                    "pros": history["aqp_pros"][-1],
                })

    # -- A-3: TSP distillation --
    log.info("Stage A-3: TSP distillation")
    tsp.train()
    teacher.eval()
    tsp_opt = _build_optimizer(tsp.parameters(), cfg.lr, cfg.weight_decay)
    tau = cfg.teacher_temperature

    for epoch in range(cfg.epochs):
        for step, batch in enumerate(train_loader):
            batch: DARTBatch = batch.to(device)
            tsp_opt.zero_grad(set_to_none=True)

            cache = cache_factory()

            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                B, T = batch.token_ids.shape
                distill_loss_accum = torch.tensor(0.0, device=device)

                for b in range(B):
                    length = int(batch.token_lens[b].item())
                    if length < 2:
                        continue

                    ids_b = batch.token_ids[b, :length]
                    positions = torch.arange(
                        length, dtype=torch.long, device=device,
                    )

                    # Student forward
                    student_out: TSPOutput = tsp(
                        ids_b[:-1], positions[:-1], cache, seg_id=0,
                    )

                    # Teacher forward (no grad)
                    with torch.no_grad():
                        teacher_out = teacher(
                            ids_b[:-1], positions[:-1], cache, seg_id=0,
                        )
                        teacher_logits = teacher_out.logits.detach()

                    # KL divergence at temperature tau
                    student_log_prob = F.log_softmax(
                        student_out.logits / tau, dim=-1,
                    )
                    teacher_prob = F.softmax(
                        teacher_logits / tau, dim=-1,
                    )
                    kl = F.kl_div(
                        student_log_prob, teacher_prob,
                        reduction="batchmean",
                    ) * (tau ** 2)

                    distill_loss_accum = distill_loss_accum + kl
                    cache.reset()

                distill_loss = distill_loss_accum / max(B, 1)

            scaler.scale(distill_loss).backward()
            scaler.unscale_(tsp_opt)
            nn.utils.clip_grad_norm_(tsp.parameters(), cfg.grad_clip)
            scaler.step(tsp_opt)
            scaler.update()

            history["tsp_distill"].append(distill_loss.item())
            if step % cfg.log_every == 0:
                _log_metrics("A-3", epoch, step, {
                    "distill": distill_loss.item(),
                })

    return history


# =====================================================================
# Stage B -- Cross-Path Alignment
# =====================================================================

def train_stage_b(
    *,
    aqp: AQPDecoder,
    tsp: TSPDecoder,
    align_loss_fn: ContrastiveAlignmentLoss,
    train_loader: DataLoader,
    cache_factory,
    cfg: StageBConfig,
    device: torch.device,
) -> dict[str, list[float]]:
    """Stage B: cross-path alignment with KV-drop regularisation.

    Both AQP and TSP are trained jointly through the shared KV cache.
    A teacher-forced forward pass through AQP produces draft hidden
    states, followed by a TSP forward to produce verified hidden
    states.  The contrastive alignment loss aligns audio and text
    embeddings while enforcing speaker consistency.

    **KV-drop** is applied to non-committed cache entries after the
    AQP write and before the TSP read.  The drop probability ramps
    linearly from 0 to ``cfg.kv_drop_max`` over the course of the
    stage.

    Parameters
    ----------
    aqp : AQPDecoder
    tsp : TSPDecoder
    align_loss_fn : ContrastiveAlignmentLoss
        Pre-constructed alignment loss (determines d_hidden, d_speaker,
        gamma, temperature).
    train_loader : DataLoader
        Must yield batches with ``z_audio``, ``z_text``, and
        ``speaker_emb`` populated.
    cache_factory : callable
        Returns a fresh :class:`SharedKVCache` per call.
    cfg : StageBConfig
    device : torch.device

    Returns
    -------
    dict[str, list[float]]
        Per-step loss and KV-drop probability history.
    """
    history: dict[str, list[float]] = {
        "align": [], "kv_drop_prob": [],
    }

    all_params = (
        list(aqp.parameters())
        + list(tsp.parameters())
        + list(align_loss_fn.parameters())
    )
    optimizer = _build_optimizer(all_params, cfg.lr, cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    aqp.train()
    tsp.train()
    align_loss_fn.train()

    total_steps = cfg.epochs * len(train_loader)
    global_step = 0

    for epoch in range(cfg.epochs):
        for step, batch in enumerate(train_loader):
            batch: DARTBatch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            # Linear ramp: drop_prob goes from 0 to kv_drop_max
            drop_prob = cfg.kv_drop_max * (global_step / max(1, total_steps - 1))

            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                B = batch.token_ids.shape[0]
                loss_accum = torch.tensor(0.0, device=device)
                valid_samples = 0

                for b in range(B):
                    length = int(batch.token_lens[b].item())
                    if length < 2:
                        continue

                    cache = cache_factory()

                    ids_b = batch.token_ids[b, :length]
                    positions = torch.arange(
                        length, dtype=torch.long, device=device,
                    )

                    # AQP teacher-forced pass (draft writes to cache)
                    aqp_out: AQPOutput = aqp(
                        ids_b[:-1], positions[:-1], cache, seg_id=0,
                    )

                    # KV-drop on non-committed entries
                    kv_drop(cache, drop_prob)

                    # TSP teacher-forced pass (reads from cache)
                    tsp_out: TSPOutput = tsp(
                        ids_b[:-1], positions[:-1], cache, seg_id=0,
                    )

                    # Use mean hidden state as the representation
                    h_combined = (aqp_out.hidden.mean(dim=0) +
                                  tsp_out.hidden.mean(dim=0)) / 2.0

                    z_ac = batch.z_audio[b].unsqueeze(0)
                    z_tx = batch.z_text[b].unsqueeze(0)
                    h_b = h_combined.unsqueeze(0)
                    s_b = batch.speaker_emb[b].unsqueeze(0)

                    loss = align_loss_fn(z_ac, z_tx, h_b, s_b)
                    loss_accum = loss_accum + loss
                    valid_samples += 1

                if valid_samples > 0:
                    total_loss = loss_accum / valid_samples
                else:
                    total_loss = loss_accum

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            history["align"].append(total_loss.item())
            history["kv_drop_prob"].append(drop_prob)

            if step % cfg.log_every == 0:
                _log_metrics("B", epoch, step, {
                    "align": total_loss.item(),
                    "kv_drop_p": drop_prob,
                })

            global_step += 1

    return history


# =====================================================================
# Stage C -- Joint End-to-End Optimisation
# =====================================================================

def train_stage_c(
    *,
    encoder: nn.Module,
    aqp: AQPDecoder,
    tsp: TSPDecoder,
    train_loader: DataLoader,
    cache_factory,
    cfg: StageCConfig,
    device: torch.device,
) -> dict[str, list[float]]:
    """Stage C: joint training under the full latency-aware loss.

    All components (encoder, AQP, TSP) are trained end-to-end.  The
    optimiser is AdamW with cosine learning-rate decay and linear
    warm-up.  Mixed-precision training via ``torch.cuda.amp`` is
    enabled by default.

    The training loop simulates the inference cycle per sample:

    1. Encode audio features into the shared cache.
    2. Teacher-forced AQP forward producing draft logits and prosody.
    3. Teacher-forced TSP forward producing verification logits.
    4. Compute all constituent losses and combine via
       :class:`~dart.losses.DARTJointLoss`.

    Parameters
    ----------
    encoder : nn.Module
        Audio encoder (an :class:`EncoderWrapper` or compatible).
        Must expose ``encoder`` (backbone) and ``ctc_proj`` (head).
    aqp : AQPDecoder
    tsp : TSPDecoder
    train_loader : DataLoader
        Yields :class:`~dart.data.DARTBatch`.
    cache_factory : callable
        Returns a fresh :class:`SharedKVCache`.
    cfg : StageCConfig
    device : torch.device

    Returns
    -------
    dict[str, list[float]]
        Per-step total loss and per-component breakdowns.
    """
    history: dict[str, list[float]] = {
        "total": [], "ctc": [], "draft": [], "pros": [], "ver": [], "lat": [],
        "lr": [],
    }

    # Gather all parameters
    all_params = (
        list(encoder.parameters())
        + list(aqp.parameters())
        + list(tsp.parameters())
    )

    optimizer = _build_optimizer(all_params, cfg.lr, cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    scheduler = _cosine_schedule_with_warmup(
        optimizer, cfg.warmup_steps, total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    joint_loss_fn = DARTJointLoss(weights=cfg.loss_weights)
    prosody_loss_fn = ProsodyLoss()
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean")

    encoder.train()
    aqp.train()
    tsp.train()

    global_step = 0

    for epoch in range(cfg.epochs):
        for step, batch in enumerate(train_loader):
            batch: DARTBatch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                B, T = batch.token_ids.shape

                # -- Encoder forward --
                enc_hidden = encoder.encoder(batch.audio_features)
                ctc_logits = encoder.ctc_proj(enc_hidden)
                log_probs = F.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)

                l_ctc = ctc_loss_fn(
                    log_probs,
                    batch.token_ids,
                    batch.audio_lens,
                    batch.token_lens,
                )

                # -- Per-sample AQP + TSP --
                draft_loss_sum = torch.tensor(0.0, device=device)
                pros_loss_sum = torch.tensor(0.0, device=device)
                ver_loss_sum = torch.tensor(0.0, device=device)
                valid_count = 0

                for b in range(B):
                    length = int(batch.token_lens[b].item())
                    if length < 2:
                        continue

                    cache = cache_factory()
                    ids_b = batch.token_ids[b, :length]
                    positions = torch.arange(
                        length, dtype=torch.long, device=device,
                    )

                    # Write encoder features into cache
                    F_frames = int(batch.audio_lens[b].item())
                    d_k = aqp.cfg.d_model // aqp.cfg.num_heads
                    enc_h = enc_hidden[b, :F_frames]  # (F, enc_dim)
                    # Project to cache shape: (F, num_heads, d_k)
                    enc_k = enc_h.unsqueeze(1).expand(
                        -1, aqp.cfg.num_heads, -1,
                    )[..., :d_k].contiguous()
                    enc_v = enc_k.clone()
                    enc_pos = torch.arange(
                        F_frames, dtype=torch.long, device=device,
                    )
                    cache.write_enc(enc_k, enc_v, enc_pos, seg_id=0)

                    # AQP teacher-forced
                    aqp_out: AQPOutput = aqp(
                        ids_b[:-1], positions[:-1], cache, seg_id=0,
                    )
                    T_out = aqp_out.logits.shape[0]
                    draft_loss_sum = draft_loss_sum + F.cross_entropy(
                        aqp_out.logits, ids_b[1:],
                    )

                    # Prosody loss
                    pros_loss_sum = pros_loss_sum + prosody_loss_fn(
                        aqp_out.duration[:T_out],
                        batch.durations[b, 1:length],
                        aqp_out.f0_logits[:T_out],
                        batch.f0_bins[b, 1:length],
                        aqp_out.energy_logits[:T_out],
                        batch.energy_bins[b, 1:length],
                    )

                    # TSP teacher-forced
                    tsp_out: TSPOutput = tsp(
                        ids_b[:-1], positions[:-1], cache, seg_id=0,
                    )
                    ver_loss_sum = ver_loss_sum + F.cross_entropy(
                        tsp_out.logits, ids_b[1:],
                    )

                    valid_count += 1

                # Average across samples
                denom = max(valid_count, 1)
                l_draft = draft_loss_sum / denom
                l_pros = pros_loss_sum / denom
                l_ver = ver_loss_sum / denom

                # Latency terms (simulated as zero during training;
                # in production these come from runtime measurements)
                ttfa = torch.tensor(0.0, device=device)
                rollback_rate = torch.tensor(0.0, device=device)

                breakdown: LossBreakdown = joint_loss_fn(
                    l_ctc, l_draft, l_pros, l_ver, ttfa, rollback_rate,
                )

            scaler.scale(breakdown.total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            history["total"].append(breakdown.total.item())
            history["ctc"].append(breakdown.ctc.item())
            history["draft"].append(breakdown.draft.item())
            history["pros"].append(breakdown.pros.item())
            history["ver"].append(breakdown.ver.item())
            history["lat"].append(breakdown.lat.item())
            history["lr"].append(current_lr)

            if step % cfg.log_every == 0:
                _log_metrics("C", epoch, step, {
                    "total": breakdown.total.item(),
                    "ctc": breakdown.ctc.item(),
                    "draft": breakdown.draft.item(),
                    "pros": breakdown.pros.item(),
                    "ver": breakdown.ver.item(),
                    "lat": breakdown.lat.item(),
                    "lr": current_lr,
                })

            global_step += 1

    return history
