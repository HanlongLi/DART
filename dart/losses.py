"""Loss functions for the DART architecture.

This module implements every loss term used during DART training:

Contrastive alignment loss
    .. math::

        \\mathcal{L}_{\\text{align}}
            = \\text{InfoNCE}(z^{ac}, z^{tx})
            + \\gamma \\|\\Pi(h) - s\\|_2^2

    where :math:`z^{ac}` and :math:`z^{tx}` are L2-normalised audio and
    text embeddings, :math:`\\Pi` is a learned projector head, :math:`h`
    is a hidden state, :math:`s` is a speaker embedding, and
    :math:`\\gamma` controls the speaker-consistency penalty.

Joint latency-aware loss
    .. math::

        \\mathcal{L}
            = \\lambda_{ctc}   \\mathcal{L}_{ctc}
            + \\lambda_{draft} \\mathcal{L}_{draft}
            + \\lambda_{pros}  \\mathcal{L}_{pros}
            + \\lambda_{ver}   \\mathcal{L}_{ver}
            + \\lambda_{lat}   \\mathcal{L}_{lat}

    with the latency penalty

    .. math::

        \\mathcal{L}_{lat}
            = \\max(0,\\, \\text{TTFA} - 250)
            + 0.5 \\cdot \\text{RollbackRate}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1.  InfoNCE  (building-block)
# =====================================================================

class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE contrastive loss.

    Given two sets of L2-normalised embeddings
    :math:`z^{ac} \\in \\mathbb{R}^{B \\times D}` and
    :math:`z^{tx} \\in \\mathbb{R}^{B \\times D}` the loss is

    .. math::

        \\ell = \\frac{1}{2}\\bigl(
            \\text{CE}(\\text{sim} / \\tau,\\; \\mathbf{I})_{\\text{rows}}
          + \\text{CE}(\\text{sim} / \\tau,\\; \\mathbf{I})_{\\text{cols}}
        \\bigr)

    where :math:`\\text{sim} = z^{ac} {z^{tx}}^\\top`.

    Parameters
    ----------
    temperature : float
        Softmax temperature :math:`\\tau`.  Learnable if
        ``learnable_temperature=True``.
    learnable_temperature : bool
        If ``True``, :math:`\\log\\tau` is registered as a parameter.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temperature: bool = False,
    ) -> None:
        super().__init__()
        if learnable_temperature:
            self.log_temperature = nn.Parameter(
                torch.tensor(temperature).log()
            )
        else:
            self.register_buffer(
                "log_temperature",
                torch.tensor(temperature).log(),
            )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(
        self,
        z_ac: torch.Tensor,
        z_tx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the symmetric InfoNCE loss.

        Parameters
        ----------
        z_ac : ``(B, D)`` -- L2-normalised audio embeddings.
        z_tx : ``(B, D)`` -- L2-normalised text embeddings.

        Returns
        -------
        torch.Tensor  scalar loss.
        """
        # Similarity matrix  (B, B)
        logits = z_ac @ z_tx.T / self.temperature   # (B, B)
        B = logits.shape[0]
        labels = torch.arange(B, device=logits.device)

        loss_ac = F.cross_entropy(logits, labels)
        loss_tx = F.cross_entropy(logits.T, labels)
        return (loss_ac + loss_tx) / 2.0


# =====================================================================
# 2.  Contrastive Alignment Loss  (L_align)
# =====================================================================

class ContrastiveAlignmentLoss(nn.Module):
    r"""Contrastive alignment with speaker-consistency regulariser.

    .. math::

        \mathcal{L}_{\text{align}}
            = \text{InfoNCE}(z^{ac}, z^{tx})
            + \gamma \|\Pi(h) - s\|_2^2

    Parameters
    ----------
    d_hidden : int
        Dimension of the hidden state *h*.
    d_speaker : int
        Dimension of the speaker embedding *s*.
    temperature : float
        InfoNCE softmax temperature.
    gamma : float
        Weight :math:`\gamma` of the speaker-consistency L2 penalty.
    learnable_temperature : bool
        Whether the InfoNCE temperature is learnable.
    """

    def __init__(
        self,
        d_hidden: int,
        d_speaker: int,
        temperature: float = 0.07,
        gamma: float = 1.0,
        learnable_temperature: bool = False,
    ) -> None:
        super().__init__()
        self.gamma = gamma

        self.info_nce = InfoNCELoss(
            temperature=temperature,
            learnable_temperature=learnable_temperature,
        )

        # Projector head  Pi : h -> speaker space
        self.projector = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_speaker),
        )

    def forward(
        self,
        z_ac: torch.Tensor,
        z_tx: torch.Tensor,
        h: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute :math:`\mathcal{L}_{\text{align}}`.

        Parameters
        ----------
        z_ac : ``(B, D)`` -- L2-normalised audio embeddings.
        z_tx : ``(B, D)`` -- L2-normalised text embeddings.
        h    : ``(B, d_hidden)`` -- hidden states fed to projector.
        s    : ``(B, d_speaker)`` -- target speaker embeddings.

        Returns
        -------
        torch.Tensor  scalar loss.
        """
        nce = self.info_nce(z_ac, z_tx)
        proj = self.projector(h)            # (B, d_speaker)
        spk = self.gamma * F.mse_loss(proj, s)
        return nce + spk


# =====================================================================
# 3.  Prosody Loss  (L_pros)
# =====================================================================

class ProsodyLoss(nn.Module):
    """Combined prosody loss for AQP predictions.

    .. math::

        \\mathcal{L}_{pros}
            = \\text{MSE}(\\hat d, d)
            + \\text{CE}(\\hat f_0, f_0)
            + \\text{CE}(\\hat e, e)

    where :math:`\\hat d` is predicted duration, :math:`\\hat f_0` are
    coarse F0 bin logits, and :math:`\\hat e` are coarse energy bin
    logits.
    """

    def forward(
        self,
        pred_duration: torch.Tensor,
        target_duration: torch.Tensor,
        pred_f0_logits: torch.Tensor,
        target_f0_bins: torch.Tensor,
        pred_energy_logits: torch.Tensor,
        target_energy_bins: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the prosody loss.

        Parameters
        ----------
        pred_duration      : ``(T, 1)``  -- predicted duration.
        target_duration    : ``(T, 1)``  -- ground-truth duration.
        pred_f0_logits     : ``(T, n_f0_bins)``  -- F0 bin logits.
        target_f0_bins     : ``(T,)``  -- ground-truth F0 bin indices.
        pred_energy_logits : ``(T, n_energy_bins)``  -- energy logits.
        target_energy_bins : ``(T,)``  -- ground-truth energy bin indices.

        Returns
        -------
        torch.Tensor  scalar.
        """
        dur_loss = F.mse_loss(pred_duration, target_duration)
        f0_loss = F.cross_entropy(pred_f0_logits, target_f0_bins)
        energy_loss = F.cross_entropy(pred_energy_logits, target_energy_bins)
        return dur_loss + f0_loss + energy_loss


# =====================================================================
# 4.  Latency Penalty  (L_lat)
# =====================================================================

class LatencyLoss(nn.Module):
    r"""Latency-aware penalty term.

    .. math::

        \mathcal{L}_{\text{lat}}
            = \max(0,\; \text{TTFA} - \text{ttfa\_target})
            + \alpha \cdot \text{RollbackRate}

    Parameters
    ----------
    ttfa_target : float
        TTFA threshold in milliseconds (default 250 ms).
    rollback_weight : float
        Multiplier :math:`\alpha` on the rollback rate (default 0.5).
    """

    def __init__(
        self,
        ttfa_target: float = 250.0,
        rollback_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.ttfa_target = ttfa_target
        self.rollback_weight = rollback_weight

    def forward(
        self,
        ttfa: torch.Tensor,
        rollback_rate: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute :math:`\mathcal{L}_{\text{lat}}`.

        Parameters
        ----------
        ttfa : scalar ``Tensor``
            Time-To-First-Audio in **ms** (may be a batch mean).
        rollback_rate : scalar ``Tensor``
            Fraction of tokens that had to be rolled back
            (:math:`\in [0, 1]`).

        Returns
        -------
        torch.Tensor  scalar loss.
        """
        ttfa_penalty = F.relu(ttfa - self.ttfa_target)
        return ttfa_penalty + self.rollback_weight * rollback_rate


# =====================================================================
# 5.  Joint Loss  (L)
# =====================================================================

@dataclass
class LossWeights:
    r"""Scalar multipliers :math:`\lambda_*` for each term.

    Attributes
    ----------
    ctc : float
        Weight on the CTC loss.
    draft : float
        Weight on the AQP draft cross-entropy loss.
    pros : float
        Weight on the prosody loss.
    ver : float
        Weight on the TSP verification cross-entropy loss.
    lat : float
        Weight on the latency penalty.
    """
    ctc: float = 1.0
    draft: float = 1.0
    pros: float = 0.5
    ver: float = 1.0
    lat: float = 0.1


@dataclass
class LossBreakdown:
    """Itemised loss values returned by :class:`DARTJointLoss`.

    Every field is a detached scalar tensor (for logging / TensorBoard).
    """
    total: torch.Tensor
    ctc: torch.Tensor
    draft: torch.Tensor
    pros: torch.Tensor
    ver: torch.Tensor
    lat: torch.Tensor


class DARTJointLoss(nn.Module):
    r"""Joint latency-aware training objective for DART.

    .. math::

        \mathcal{L}
            = \lambda_{ctc}   \mathcal{L}_{ctc}
            + \lambda_{draft} \mathcal{L}_{draft}
            + \lambda_{pros}  \mathcal{L}_{pros}
            + \lambda_{ver}   \mathcal{L}_{ver}
            + \lambda_{lat}   \mathcal{L}_{lat}

    Each constituent loss is computed **externally** and passed into
    :meth:`forward` so that the caller retains full flexibility over
    data flow.  This module simply applies the weights, sums, and
    returns a :class:`LossBreakdown`.

    The :math:`\mathcal{L}_{lat}` term is computed internally from raw
    TTFA and rollback-rate scalars via :class:`LatencyLoss`.

    Parameters
    ----------
    weights : LossWeights
        Per-term :math:`\lambda` multipliers.
    ttfa_target : float
        Passed to :class:`LatencyLoss`.
    rollback_weight : float
        Passed to :class:`LatencyLoss`.
    """

    def __init__(
        self,
        weights: Optional[LossWeights] = None,
        ttfa_target: float = 250.0,
        rollback_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.w = weights or LossWeights()
        self.latency_loss = LatencyLoss(
            ttfa_target=ttfa_target,
            rollback_weight=rollback_weight,
        )

    def forward(
        self,
        l_ctc: torch.Tensor,
        l_draft: torch.Tensor,
        l_pros: torch.Tensor,
        l_ver: torch.Tensor,
        ttfa: torch.Tensor,
        rollback_rate: torch.Tensor,
    ) -> LossBreakdown:
        r"""Compute the joint loss and return an itemised breakdown.

        All input losses should be **scalar tensors** with gradients
        attached (except ``ttfa`` and ``rollback_rate`` which may be
        detached runtime measurements).

        Parameters
        ----------
        l_ctc : torch.Tensor
            CTC alignment loss (encoder -> text).
        l_draft : torch.Tensor
            AQP draft cross-entropy loss.
        l_pros : torch.Tensor
            Prosody prediction loss (duration + F0 + energy).
        l_ver : torch.Tensor
            TSP verification cross-entropy loss.
        ttfa : torch.Tensor
            Time-To-First-Audio in ms (scalar).
        rollback_rate : torch.Tensor
            Rollback rate in [0, 1] (scalar).

        Returns
        -------
        LossBreakdown
            ``.total`` is the weighted sum (with grad); all other
            fields are ``.detach()``-ed copies for logging.
        """
        l_lat = self.latency_loss(ttfa, rollback_rate)

        total = (
            self.w.ctc   * l_ctc
            + self.w.draft * l_draft
            + self.w.pros  * l_pros
            + self.w.ver   * l_ver
            + self.w.lat   * l_lat
        )

        return LossBreakdown(
            total=total,
            ctc=l_ctc.detach(),
            draft=l_draft.detach(),
            pros=l_pros.detach(),
            ver=l_ver.detach(),
            lat=l_lat.detach(),
        )
