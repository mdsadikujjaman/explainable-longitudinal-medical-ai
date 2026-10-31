"""NACC longitudinal multimodal progression modeling utilities."""

from nacc_progression.contracts import PatientBatch, PatientSequence, PredictionResult

try:
    from nacc_progression.model import ModelConfig, MultimodalGRU
except ImportError:
    ModelConfig = None
    MultimodalGRU = None

__all__ = [
    "ModelConfig",
    "MultimodalGRU",
    "PatientBatch",
    "PatientSequence",
    "PredictionResult",
]
