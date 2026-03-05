"""DART -- Dual-Path Audio-Reasoning Transformer.

Core modules for the DART streaming speech-to-speech architecture.
"""

from dart.shared_kv_cache import SharedKVCache, Source, CacheReadOut
from dart.layers import CacheAwareMHA, CacheAwareTransformerBlock, RotaryEmbedding
from dart.aqp_decoder import AQPDecoder, AQPConfig, AQPOutput
from dart.tsp_decoder import TSPDecoder, TSPConfig, TSPOutput
from dart.engine import DARTInferenceEngine, EngineConfig, StepResult
from dart.losses import (
    InfoNCELoss,
    ContrastiveAlignmentLoss,
    ProsodyLoss,
    LatencyLoss,
    DARTJointLoss,
    LossWeights,
    LossBreakdown,
)
from dart.data import (
    DARTSample,
    DARTBatch,
    DARTDataset,
    dart_collate_fn,
    build_dataloader,
    make_synthetic_samples,
)
from dart.training import (
    StageAConfig,
    StageBConfig,
    StageCConfig,
    EncoderWrapper,
    kv_drop,
    train_stage_a,
    train_stage_b,
    train_stage_c,
)

__all__ = [
    # Phase 1 -- shared cache
    "SharedKVCache", "Source", "CacheReadOut",
    # Phase 2 -- layers
    "RotaryEmbedding", "CacheAwareMHA", "CacheAwareTransformerBlock",
    # Phase 2 -- decoders
    "AQPDecoder", "AQPConfig", "AQPOutput",
    "TSPDecoder", "TSPConfig", "TSPOutput",
    # Phase 3 -- inference engine
    "DARTInferenceEngine", "EngineConfig", "StepResult",
    # Phase 4 -- losses
    "InfoNCELoss", "ContrastiveAlignmentLoss", "ProsodyLoss",
    "LatencyLoss", "DARTJointLoss", "LossWeights", "LossBreakdown",
    # Phase 5 -- data
    "DARTSample", "DARTBatch", "DARTDataset",
    "dart_collate_fn", "build_dataloader", "make_synthetic_samples",
    # Phase 5 -- training
    "StageAConfig", "StageBConfig", "StageCConfig",
    "EncoderWrapper", "kv_drop",
    "train_stage_a", "train_stage_b", "train_stage_c",
]
__version__ = "0.1.0"
