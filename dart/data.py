"""Dataset and collation utilities for DART training.

Each training sample is a paired utterance consisting of:

* ``audio_features`` -- encoder hidden states extracted from the audio
  front-end, shape ``(F, enc_dim)`` where *F* varies across samples.
* ``token_ids`` -- ground-truth token-id sequence, shape ``(T,)``.
* ``durations`` -- per-token duration targets, shape ``(T, 1)``.
* ``f0_bins`` -- coarse F0 bin indices, shape ``(T,)``.
* ``energy_bins`` -- coarse energy bin indices, shape ``(T,)``.
* ``speaker_emb`` -- speaker embedding, shape ``(d_speaker,)``.

For contrastive alignment (Stage B) the dataset also yields:

* ``z_audio`` -- L2-normalised audio embedding, ``(D,)``.
* ``z_text`` -- L2-normalised text embedding, ``(D,)``.

All tensors are expected to reside on CPU; the training loop moves
them to the target device.  Variable-length fields are padded by
:func:`dart_collate_fn`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


# =====================================================================
# Sample container
# =====================================================================

@dataclass
class DARTSample:
    """A single training example.

    Fields set to ``None`` are not required by every training stage.
    """
    audio_features: torch.Tensor          # (F, enc_dim)
    token_ids: torch.Tensor               # (T,)
    durations: torch.Tensor               # (T, 1)
    f0_bins: torch.Tensor                 # (T,)
    energy_bins: torch.Tensor             # (T,)
    speaker_emb: torch.Tensor             # (d_speaker,)
    z_audio: Optional[torch.Tensor] = None   # (D,)
    z_text: Optional[torch.Tensor] = None    # (D,)


# =====================================================================
# Collated batch
# =====================================================================

@dataclass
class DARTBatch:
    """Padded, collated batch ready for a forward pass.

    ``*_lens`` tensors record the original (unpadded) lengths so that
    loss functions can apply masking where needed.

    Attributes
    ----------
    audio_features : ``(B, F_max, enc_dim)``
    audio_lens : ``(B,)``
    token_ids : ``(B, T_max)``
    token_lens : ``(B,)``
    durations : ``(B, T_max, 1)``
    f0_bins : ``(B, T_max)``
    energy_bins : ``(B, T_max)``
    speaker_emb : ``(B, d_speaker)``
    z_audio : ``(B, D)`` or ``None``
    z_text : ``(B, D)`` or ``None``
    """
    audio_features: torch.Tensor
    audio_lens: torch.Tensor
    token_ids: torch.Tensor
    token_lens: torch.Tensor
    durations: torch.Tensor
    f0_bins: torch.Tensor
    energy_bins: torch.Tensor
    speaker_emb: torch.Tensor
    z_audio: Optional[torch.Tensor] = None
    z_text: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "DARTBatch":
        """Move every tensor field to *device*."""
        kwargs: dict = {}
        for name in self.__dataclass_fields__:
            val = getattr(self, name)
            if isinstance(val, torch.Tensor):
                kwargs[name] = val.to(device)
            else:
                kwargs[name] = val
        return DARTBatch(**kwargs)


# =====================================================================
# Collate function
# =====================================================================

def dart_collate_fn(samples: list[DARTSample]) -> DARTBatch:
    """Pad and collate a list of :class:`DARTSample` into a batch.

    Audio features are padded along the frame axis, token-level fields
    along the token axis.  Padding value is ``0`` for ids / bins and
    ``0.0`` for continuous features.
    """
    audio_lens = torch.tensor(
        [s.audio_features.shape[0] for s in samples], dtype=torch.long,
    )
    token_lens = torch.tensor(
        [s.token_ids.shape[0] for s in samples], dtype=torch.long,
    )

    audio_features = pad_sequence(
        [s.audio_features for s in samples], batch_first=True,
    )
    token_ids = pad_sequence(
        [s.token_ids for s in samples], batch_first=True, padding_value=0,
    )
    durations = pad_sequence(
        [s.durations for s in samples], batch_first=True,
    )
    f0_bins = pad_sequence(
        [s.f0_bins for s in samples], batch_first=True, padding_value=0,
    )
    energy_bins = pad_sequence(
        [s.energy_bins for s in samples], batch_first=True, padding_value=0,
    )
    speaker_emb = torch.stack([s.speaker_emb for s in samples])

    has_align = all(s.z_audio is not None and s.z_text is not None for s in samples)
    z_audio = torch.stack([s.z_audio for s in samples]) if has_align else None
    z_text = torch.stack([s.z_text for s in samples]) if has_align else None

    return DARTBatch(
        audio_features=audio_features,
        audio_lens=audio_lens,
        token_ids=token_ids,
        token_lens=token_lens,
        durations=durations,
        f0_bins=f0_bins,
        energy_bins=energy_bins,
        speaker_emb=speaker_emb,
        z_audio=z_audio,
        z_text=z_text,
    )


# =====================================================================
# Synthetic / placeholder dataset
# =====================================================================

class DARTDataset(Dataset):
    """In-memory dataset backed by a list of :class:`DARTSample`.

    For real training, subclass and override :meth:`__getitem__` to
    load features from disk.

    Parameters
    ----------
    samples : list[DARTSample]
        Pre-built samples (typically synthetic for unit tests).
    """

    def __init__(self, samples: list[DARTSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DARTSample:
        return self.samples[idx]


# =====================================================================
# DataLoader factory
# =====================================================================

def build_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
) -> DataLoader:
    """Convenience builder that wires in :func:`dart_collate_fn`.

    Parameters
    ----------
    dataset : Dataset
        Must yield :class:`DARTSample` instances.
    batch_size : int
    shuffle : bool
    num_workers : int
        Number of data-loading worker processes.  Set to > 0 on a
        multi-GPU lab server with fast storage.
    pin_memory : bool
        Set ``True`` when training on CUDA for faster host-to-device
        transfers.
    drop_last : bool
        Drop the last incomplete batch to keep shapes consistent.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=dart_collate_fn,
    )


# =====================================================================
# Synthetic data factory (for testing / smoke-runs)
# =====================================================================

def make_synthetic_samples(
    n: int = 64,
    enc_dim: int = 256,
    vocab_size: int = 1024,
    n_f0_bins: int = 64,
    n_energy_bins: int = 32,
    d_speaker: int = 64,
    d_align: int = 128,
    max_frames: int = 200,
    max_tokens: int = 60,
    include_alignment: bool = False,
    seed: int = 0,
) -> list[DARTSample]:
    """Generate *n* random :class:`DARTSample` instances.

    Useful for integration tests and smoke-run sanity checks before
    launching on real data.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    enc_dim : int
        Encoder hidden dimension.
    vocab_size : int
        Token vocabulary size.
    n_f0_bins, n_energy_bins : int
        Bin counts for prosody targets.
    d_speaker : int
        Speaker embedding dimension.
    d_align : int
        Alignment embedding dimension (for z_audio / z_text).
    max_frames : int
        Upper bound on randomly sampled frame count.
    max_tokens : int
        Upper bound on randomly sampled token count.
    include_alignment : bool
        If ``True``, populate ``z_audio`` and ``z_text``.
    seed : int
        Random seed for reproducibility.
    """
    rng = torch.Generator().manual_seed(seed)

    samples: list[DARTSample] = []
    for _ in range(n):
        F_frames = int(torch.randint(10, max_frames + 1, (1,), generator=rng).item())
        T_tokens = int(torch.randint(5, max_tokens + 1, (1,), generator=rng).item())

        audio = torch.randn(F_frames, enc_dim, generator=rng)
        tids = torch.randint(0, vocab_size, (T_tokens,), generator=rng)
        dur = torch.randn(T_tokens, 1, generator=rng).abs()
        f0 = torch.randint(0, n_f0_bins, (T_tokens,), generator=rng)
        energy = torch.randint(0, n_energy_bins, (T_tokens,), generator=rng)
        spk = torch.randn(d_speaker, generator=rng)

        z_a = z_t = None
        if include_alignment:
            z_a = torch.randn(d_align, generator=rng)
            z_a = z_a / z_a.norm()
            z_t = torch.randn(d_align, generator=rng)
            z_t = z_t / z_t.norm()

        samples.append(DARTSample(
            audio_features=audio,
            token_ids=tids,
            durations=dur,
            f0_bins=f0,
            energy_bins=energy,
            speaker_emb=spk,
            z_audio=z_a,
            z_text=z_t,
        ))

    return samples
