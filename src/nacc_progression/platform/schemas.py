"""API-facing Pydantic schemas and serialization layer for Workstream 5.

Provides JSON-serializable request/response models and bidirectional converters
connecting the FastAPI/Streamlit layers with the scientific contracts.
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from nacc_progression.contracts import (
    CLINICAL_FEATURE_COUNT,
    MAX_VISITS,
    PatientSequence,
    PredictionResult,
)

try:
    from nacc_progression.explainability import ExplanationResult, VisitAblationResult
except ImportError:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class ExplanationResult:
        subject_key: str
        probability: float
        mri_attributions: np.ndarray
        clinical_attributions: np.ndarray
        delta_time_attributions: np.ndarray
        mri_feature_importance: np.ndarray
        clinical_feature_importance: np.ndarray
        visit_importance: np.ndarray
        modality_importance: dict[str, float]
        completeness_delta_logit: float
        mri_feature_names: tuple[str, ...]
        clinical_feature_names: tuple[str, ...]
        baseline_description: str = (
            "Zero normalized values (training mean after z-scoring); observed/missing "
            "masks are held fixed."
        )

    @dataclass(frozen=True)
    class VisitAblationResult:
        removed_visit_index: int
        probability_without_visit: float
        signed_probability_change: float
        absolute_probability_change: float
        remaining_visit_count: int


CLINICAL_ORDER = ("age", "sex", "education", "cdr_global", "cdr_sum_boxes", "mmse")


# ============================================================================
# 1. Authentication Schemas
# ============================================================================

class LoginRequest(BaseModel):
    """User credentials submitted to /api/auth/login."""
    username: str = Field(..., min_length=1, description="Username or research identifier")
    password: str = Field(..., min_length=1, description="Plaintext password")


class TokenResponse(BaseModel):
    """Authentication bearer token returned on successful login."""
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str = "researcher"


class UserInfo(BaseModel):
    """Authenticated user profile."""
    username: str
    role: str = "researcher"
    is_active: bool = True


# ============================================================================
# 2. Patient & Visit Schemas
# ============================================================================

class VisitData(BaseModel):
    """Preprocessed observations for one longitudinal visit."""
    imaging_values: list[float] = Field(..., description="Normalized MRI regional features")
    imaging_missing_mask: list[bool] = Field(..., description="True if MRI feature was missing/imputed")
    clinical_values: list[float] = Field(..., description="6 clinical features in canonical order")
    clinical_missing_mask: list[bool] = Field(..., description="6 missingness indicators for clinical features")
    delta_time_years: float = Field(..., ge=0.0, description="Elapsed time in years since previous visit")

    @field_validator("clinical_values", "clinical_missing_mask")
    @classmethod
    def check_clinical_length(cls, values: list[Any]) -> list[Any]:
        if len(values) != CLINICAL_FEATURE_COUNT:
            raise ValueError(f"Clinical features must have exactly {CLINICAL_FEATURE_COUNT} values")
        return values

    @model_validator(mode="after")
    def check_mask_lengths(self) -> VisitData:
        if len(self.imaging_values) != len(self.imaging_missing_mask):
            raise ValueError("imaging_values and imaging_missing_mask must have identical lengths")
        if len(self.clinical_values) != len(self.clinical_missing_mask):
            raise ValueError("clinical_values and clinical_missing_mask must have identical lengths")
        return self


class PatientSummary(BaseModel):
    """Summary record for patient directory and search table."""
    subject_key: str
    visit_count: int
    label: int | None = None
    has_mri: bool = True
    split: str | None = None
    age_at_anchor: float | None = None
    latest_mmse: float | None = None
    latest_cdr_global: float | None = None


class PatientDetailResponse(BaseModel):
    """Complete preprocessed history for one patient."""
    summary: PatientSummary
    visits: list[VisitData]
    imaging_feature_names: list[str] = Field(default_factory=list)
    clinical_feature_names: list[str] = Field(default_factory=lambda: list(CLINICAL_ORDER))


# ============================================================================
# 3. Prediction Schemas
# ============================================================================

class PredictionRequest(BaseModel):
    """Payload to trigger model inference on a patient sequence."""
    subject_key: str = Field(..., min_length=1)
    visits: list[VisitData] = Field(..., min_length=1, max_length=MAX_VISITS)
    imaging_feature_names: list[str] = Field(default_factory=list)
    threshold: float = Field(0.5, ge=0.0, le=1.0)

    @field_validator("visits")
    @classmethod
    def check_sequence_validity(cls, visits: list[VisitData]) -> list[VisitData]:
        if visits[0].delta_time_years != 0.0:
            raise ValueError("First visit delta_time_years must be 0.0")
        feature_count = len(visits[0].imaging_values)
        for idx, visit in enumerate(visits):
            if len(visit.imaging_values) != feature_count:
                raise ValueError(
                    f"Visit {idx} MRI feature count ({len(visit.imaging_values)}) "
                    f"does not match visit 0 ({feature_count})"
                )
        return visits


class PredictionResponse(BaseModel):
    """Progression prediction and prefix risk trajectory."""
    subject_key: str
    probability: float
    threshold: float
    predicted_label: int
    prefix_probabilities: list[float]
    prefix_semantics: str
    visit_count: int


# ============================================================================
# 4. Explainability Schemas
# ============================================================================

class FeatureAttribution(BaseModel):
    """Named feature attribution importance."""
    feature_name: str
    importance: float


class VisitAblation(BaseModel):
    """Diagnostic outcome of omitting one visit from the history."""
    removed_visit_index: int
    probability_without_visit: float
    signed_probability_change: float
    absolute_probability_change: float
    remaining_visit_count: int


class ExplanationRequest(BaseModel):
    """Payload to request Integrated Gradients explanation and ablation."""
    subject_key: str = Field(..., min_length=1)
    visits: list[VisitData] = Field(..., min_length=1, max_length=MAX_VISITS)
    imaging_feature_names: list[str] = Field(default_factory=list)
    clinical_feature_names: list[str] = Field(default_factory=lambda: list(CLINICAL_ORDER))
    steps: int = Field(32, ge=2, le=256)
    include_visit_ablation: bool = True
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class ExplanationResponse(BaseModel):
    """Detailed multi-level attribution report."""
    subject_key: str
    probability: float
    mri_attributions: list[list[float]]
    clinical_attributions: list[list[float]]
    delta_time_attributions: list[list[float]]
    mri_feature_importance: list[FeatureAttribution]
    clinical_feature_importance: list[FeatureAttribution]
    visit_importance: list[float]
    modality_importance: dict[str, float]
    completeness_delta_logit: float
    baseline_description: str
    mri_feature_names: list[str]
    clinical_feature_names: list[str]
    visit_ablation: list[VisitAblation] | None = None


# ============================================================================
# 5. Metadata and Audit Schemas
# ============================================================================

class ModelInfoResponse(BaseModel):
    """Model status, architecture, and feature definitions."""
    model_loaded: bool
    is_demo_mode: bool
    imaging_feature_count: int
    clinical_feature_count: int = CLINICAL_FEATURE_COUNT
    temporal_hidden: int = 64
    imaging_feature_names: list[str] = Field(default_factory=list)
    clinical_feature_names: list[str] = Field(default_factory=lambda: list(CLINICAL_ORDER))
    recommended_threshold: float = 0.5


class AuditResponse(BaseModel):
    """Cohort statistics, data distribution, and evaluation audit."""
    source_freeze: str
    cohort_counts: dict[str, Any]
    split_counts: dict[str, int]
    evaluation_metrics: dict[str, float] | None = None
    data_protection_warning: str


# ============================================================================
# 6. Bidirectional Conversion Functions
# ============================================================================

def request_to_patient_sequence(
    request: PredictionRequest | ExplanationRequest,
    label_placeholder: int = 0,
) -> PatientSequence:
    """Convert an API request into a strict scientific PatientSequence."""
    imaging_vals = np.array([v.imaging_values for v in request.visits], dtype=np.float32)
    imaging_masks = np.array([v.imaging_missing_mask for v in request.visits], dtype=np.bool_)
    clinical_vals = np.array([v.clinical_values for v in request.visits], dtype=np.float32)
    clinical_masks = np.array([v.clinical_missing_mask for v in request.visits], dtype=np.bool_)
    delta_time = np.array([[v.delta_time_years] for v in request.visits], dtype=np.float32)

    return PatientSequence(
        subject_key=request.subject_key,
        label=label_placeholder,
        imaging_values=imaging_vals,
        imaging_missing_mask=imaging_masks,
        clinical_values=clinical_vals,
        clinical_missing_mask=clinical_masks,
        delta_time=delta_time,
        imaging_feature_names=tuple(request.imaging_feature_names),
    )


def patient_sequence_to_visits(sequence: PatientSequence) -> list[VisitData]:
    """Convert a PatientSequence into a list of API VisitData objects."""
    visits: list[VisitData] = []
    for i in range(sequence.visit_count):
        visits.append(
            VisitData(
                imaging_values=sequence.imaging_values[i].tolist(),
                imaging_missing_mask=sequence.imaging_missing_mask[i].tolist(),
                clinical_values=sequence.clinical_values[i].tolist(),
                clinical_missing_mask=sequence.clinical_missing_mask[i].tolist(),
                delta_time_years=float(sequence.delta_time[i, 0]),
            )
        )
    return visits


def prediction_result_to_response(result: PredictionResult) -> PredictionResponse:
    """Convert an internal PredictionResult into an API PredictionResponse."""
    return PredictionResponse(
        subject_key=result.subject_key,
        probability=float(result.probability),
        threshold=float(result.threshold),
        predicted_label=int(result.predicted_label),
        prefix_probabilities=[float(p) for p in result.prefix_probabilities],
        prefix_semantics=str(result.prefix_semantics),
        visit_count=len(result.prefix_probabilities),
    )


def explanation_result_to_response(
    result: ExplanationResult,
    ablation: Sequence[VisitAblationResult] | None = None,
) -> ExplanationResponse:
    """Convert an ExplanationResult and optional ablations into an API ExplanationResponse."""
    mri_names = list(result.mri_feature_names)
    clinical_names = list(result.clinical_feature_names)

    mri_importance = [
        FeatureAttribution(feature_name=mri_names[i], importance=float(result.mri_feature_importance[i]))
        for i in range(min(len(mri_names), len(result.mri_feature_importance)))
    ]
    mri_importance.sort(key=lambda x: x.importance, reverse=True)

    clinical_importance = [
        FeatureAttribution(
            feature_name=clinical_names[i],
            importance=float(result.clinical_feature_importance[i]),
        )
        for i in range(min(len(clinical_names), len(result.clinical_feature_importance)))
    ]
    clinical_importance.sort(key=lambda x: x.importance, reverse=True)

    ablation_items: list[VisitAblation] | None = None
    if ablation is not None:
        ablation_items = [
            VisitAblation(
                removed_visit_index=item.removed_visit_index,
                probability_without_visit=float(item.probability_without_visit),
                signed_probability_change=float(item.signed_probability_change),
                absolute_probability_change=float(item.absolute_probability_change),
                remaining_visit_count=item.remaining_visit_count,
            )
            for item in ablation
        ]

    return ExplanationResponse(
        subject_key=result.subject_key,
        probability=float(result.probability),
        mri_attributions=result.mri_attributions.tolist(),
        clinical_attributions=result.clinical_attributions.tolist(),
        delta_time_attributions=result.delta_time_attributions.tolist(),
        mri_feature_importance=mri_importance,
        clinical_feature_importance=clinical_importance,
        visit_importance=result.visit_importance.tolist(),
        modality_importance={k: float(v) for k, v in result.modality_importance.items()},
        completeness_delta_logit=float(result.completeness_delta_logit),
        baseline_description=str(result.baseline_description),
        mri_feature_names=mri_names,
        clinical_feature_names=clinical_names,
        visit_ablation=ablation_items,
    )
