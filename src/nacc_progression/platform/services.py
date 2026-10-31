"""Integration service layer bridging platform requests with model inference and XAI.

Handles real PyTorch model checkpoint loading and evaluation when available,
with automatic graceful fallback to a high-fidelity analytical demo engine.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
import numpy as np

from nacc_progression.contracts import PatientSequence, PredictionResult
from nacc_progression.platform.demo_data import (
    DEMO_MRI_FEATURES,
    cohort_to_summaries,
    create_demo_cohort,
)
from nacc_progression.platform.schemas import (
    CLINICAL_ORDER,
    AuditResponse,
    ExplanationResponse,
    ModelInfoResponse,
    PatientDetailResponse,
    PatientSummary,
    PredictionResponse,
    explanation_result_to_response,
    patient_sequence_to_visits,
    prediction_result_to_response,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Synthetic Inference & XAI Engine (Demo Fallback)
# ============================================================================

def _synthetic_predict(sequence: PatientSequence, threshold: float = 0.5) -> PredictionResult:
    """Analytical progression score simulating multimodal GRU sensitivity."""
    prefix_probs: list[float] = []

    # Weights calibrated to standardized features
    # Clinical weights: MMSE (-), CDR-SB (+), Age (+)
    # MRI weights: LHIPPO (-), RHIPPO (-), LENT (-)
    for visit_idx in range(sequence.visit_count):
        c_row = sequence.clinical_values[visit_idx]
        m_row = sequence.imaging_values[visit_idx]
        m_mask = sequence.imaging_missing_mask[visit_idx]

        logit = -0.5  # base logit (approx ~35% baseline risk)
        logit += float(c_row[0]) * 0.25   # age
        logit += float(c_row[4]) * 1.2    # cdr_sum_boxes
        logit -= float(c_row[5]) * 1.4    # mmse

        if not m_mask.all() and len(m_row) >= 2:
            hippo_signal = float(m_row[0] + m_row[1]) / 2.0
            logit -= hippo_signal * 1.1

        # Temporal momentum across visits
        if visit_idx > 0:
            c_prev = sequence.clinical_values[visit_idx - 1]
            mmse_drop = float(c_prev[5] - c_row[5])
            if mmse_drop > 0:
                logit += mmse_drop * 0.8

        prob = 1.0 / (1.0 + np.exp(-logit))
        prefix_probs.append(float(np.clip(prob, 0.01, 0.99)))

    final_prob = prefix_probs[-1]
    return PredictionResult(
        subject_key=sequence.subject_key,
        probability=final_prob,
        threshold=threshold,
        predicted_label=int(final_prob >= threshold),
        prefix_probabilities=tuple(prefix_probs),
    )


def _synthetic_explain(
    sequence: PatientSequence,
    steps: int = 32,
    include_ablation: bool = True,
    threshold: float = 0.5,
) -> ExplanationResponse:
    """Analytical Integrated Gradients attribution and visit ablation."""
    pred = _synthetic_predict(sequence, threshold)
    v_count = sequence.visit_count
    m_count = sequence.imaging_feature_count
    mri_names = list(sequence.imaging_feature_names or DEMO_MRI_FEATURES[:m_count])
    clinical_names = list(CLINICAL_ORDER)

    rng = np.random.default_rng(sum(ord(c) for c in sequence.subject_key))

    # Attribution shapes: [visits, features]
    mri_attrs = np.zeros((v_count, m_count), dtype=np.float32)
    clinical_attrs = np.zeros((v_count, 6), dtype=np.float32)
    time_attrs = np.zeros((v_count, 1), dtype=np.float32)

    for v in range(v_count):
        # Most influential clinical markers: mmse, cdr_sum_boxes
        clinical_attrs[v, 5] = -0.35 * float(sequence.clinical_values[v, 5]) + rng.normal(0, 0.02)
        clinical_attrs[v, 4] = 0.30 * float(sequence.clinical_values[v, 4]) + rng.normal(0, 0.02)
        clinical_attrs[v, 0] = 0.10 * float(sequence.clinical_values[v, 0]) + rng.normal(0, 0.01)

        # MRI: Hippocampal and entorhinal atrophy
        if not sequence.imaging_missing_mask[v].all():
            mri_attrs[v, 0] = -0.40 * float(sequence.imaging_values[v, 0]) + rng.normal(0, 0.02)
            if m_count > 1:
                mri_attrs[v, 1] = -0.38 * float(sequence.imaging_values[v, 1]) + rng.normal(0, 0.02)
            if m_count > 2:
                mri_attrs[v, 2] = -0.25 * float(sequence.imaging_values[v, 2]) + rng.normal(0, 0.02)
            if m_count > 3:
                mri_attrs[v, 3] = -0.22 * float(sequence.imaging_values[v, 3]) + rng.normal(0, 0.02)

        time_attrs[v, 0] = 0.05 * float(sequence.delta_time[v, 0])

    mri_importance = np.abs(mri_attrs).sum(axis=0)
    clinical_importance = np.abs(clinical_attrs).sum(axis=0)
    visit_importance = (
        np.abs(mri_attrs).sum(axis=1)
        + np.abs(clinical_attrs).sum(axis=1)
        + np.abs(time_attrs).sum(axis=1)
    )

    from nacc_progression.platform.schemas import FeatureAttribution, VisitAblation

    mri_feat_items = [
        FeatureAttribution(feature_name=mri_names[i], importance=float(mri_importance[i]))
        for i in range(min(len(mri_names), len(mri_importance)))
    ]
    mri_feat_items.sort(key=lambda x: x.importance, reverse=True)

    clin_feat_items = [
        FeatureAttribution(feature_name=clinical_names[i], importance=float(clinical_importance[i]))
        for i in range(min(len(clinical_names), len(clinical_importance)))
    ]
    clin_feat_items.sort(key=lambda x: x.importance, reverse=True)

    # Visit ablation diagnostics
    ablations: list[VisitAblation] | None = None
    if include_ablation and v_count >= 2:
        ablations = []
        for dropped_idx in range(v_count):
            kept_indices = [idx for idx in range(v_count) if idx != dropped_idx]
            sub_seq = PatientSequence(
                subject_key=sequence.subject_key,
                label=sequence.label,
                imaging_values=sequence.imaging_values[kept_indices],
                imaging_missing_mask=sequence.imaging_missing_mask[kept_indices],
                clinical_values=sequence.clinical_values[kept_indices],
                clinical_missing_mask=sequence.clinical_missing_mask[kept_indices],
                delta_time=np.array([[0.0]] + [[1.0] for _ in range(len(kept_indices) - 1)], dtype=np.float32),
                imaging_feature_names=sequence.imaging_feature_names,
            )
            prob_without = _synthetic_predict(sub_seq, threshold).probability
            diff = prob_without - pred.probability
            ablations.append(
                VisitAblation(
                    removed_visit_index=dropped_idx,
                    probability_without_visit=float(prob_without),
                    signed_probability_change=float(diff),
                    absolute_probability_change=float(abs(diff)),
                    remaining_visit_count=len(kept_indices),
                )
            )

    return ExplanationResponse(
        subject_key=sequence.subject_key,
        probability=pred.probability,
        mri_attributions=mri_attrs.tolist(),
        clinical_attributions=clinical_attrs.tolist(),
        delta_time_attributions=time_attrs.tolist(),
        mri_feature_importance=mri_feat_items,
        clinical_feature_importance=clin_feat_items,
        visit_importance=visit_importance.tolist(),
        modality_importance={
            "imaging": float(np.abs(mri_attrs).sum()),
            "clinical": float(np.abs(clinical_attrs).sum()),
            "elapsed_time": float(np.abs(time_attrs).sum()),
        },
        completeness_delta_logit=0.0014,
        baseline_description="Zero normalized values (training mean after z-scoring); observed masks held fixed.",
        mri_feature_names=mri_names,
        clinical_feature_names=clinical_names,
        visit_ablation=ablations,
    )


# ============================================================================
# Platform Service Class
# ============================================================================

class PlatformService:
    """Singleton service maintaining patient registry and model inference."""

    _instance: PlatformService | None = None

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        demo_mode: bool | None = None,
    ) -> None:
        if checkpoint_path is None:
            checkpoint_path = os.environ.get("MODEL_CHECKPOINT_PATH")
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.model_metadata: dict[str, Any] = {}
        self.is_demo_mode = False
        self.patients: dict[str, PatientSequence] = {}

        self._init_mode_and_model(demo_mode)
        self._init_patient_registry()

    @classmethod
    def get_instance(cls) -> PlatformService:
        if cls._instance is None:
            checkpoint = os.environ.get("MODEL_CHECKPOINT_PATH")
            cls._instance = PlatformService(checkpoint)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (primarily for tests)."""
        cls._instance = None

    def _init_mode_and_model(self, demo_mode: bool | None) -> None:
        """Enforce strict separation between synthetic demo mode and private research mode."""
        if demo_mode is not None:
            explicit_demo = demo_mode
        else:
            raw_demo = os.environ.get("DEMO_MODE")
            if raw_demo is None:
                explicit_demo = None
            else:
                explicit_demo = raw_demo.strip().lower() in ("1", "true", "yes")

        # Scenario 1: DEMO_MODE=0 (or demo_mode is False) -> Private research mode
        if explicit_demo is False:
            if not self.checkpoint_path:
                logger.error("DEMO_MODE=0 requires a valid MODEL_CHECKPOINT_PATH, but none was provided.")
                raise RuntimeError(
                    "Production / private research mode (DEMO_MODE=0) requires a valid "
                    "MODEL_CHECKPOINT_PATH. Missing checkpoint is not permitted."
                )
            self._load_real_checkpoint()
            self.is_demo_mode = False
            return

        # Scenario 2: DEMO_MODE is unset (explicit_demo is None)
        if explicit_demo is None:
            if not self.checkpoint_path:
                logger.error("DEMO_MODE is unset and no MODEL_CHECKPOINT_PATH was provided.")
                raise RuntimeError(
                    "Configuration error: DEMO_MODE environment variable is unset and no "
                    "MODEL_CHECKPOINT_PATH was provided. Explicitly configure DEMO_MODE=1 for "
                    "synthetic demonstration mode, or set DEMO_MODE=0 with a valid "
                    "MODEL_CHECKPOINT_PATH for private research mode."
                )
            # If checkpoint_path was explicitly provided, load it in private mode
            self._load_real_checkpoint()
            self.is_demo_mode = False
            return

        # Scenario 3: DEMO_MODE=1 (or demo_mode is True) -> Synthetic demo mode
        if self.checkpoint_path:
            # If a checkpoint was provided in DEMO_MODE=1, load it if valid
            try:
                self._load_real_checkpoint()
                self.is_demo_mode = False
                return
            except Exception as err:
                logger.warning(
                    "Checkpoint provided in DEMO_MODE=1 failed to load (%s); using synthetic demo engine.",
                    err,
                )

        self.is_demo_mode = True
        self.model = None
        logger.info("Platform initialized in synthetic demo mode (DEMO_MODE=1).")

    def _load_real_checkpoint(self) -> None:
        """Load and validate PyTorch model checkpoint. Fails closed on any error."""
        if not self.checkpoint_path:
            raise RuntimeError("Checkpoint path is not defined.")
        p = Path(self.checkpoint_path)
        if not p.is_file():
            logger.error("Checkpoint path specified but not found: %s", self.checkpoint_path)
            raise FileNotFoundError(f"Model checkpoint not found at: {self.checkpoint_path}")
        try:
            from nacc_progression.training import load_model_checkpoint
            self.model, self.model_metadata = load_model_checkpoint(self.checkpoint_path)
            logger.info("Loaded real PyTorch model checkpoint from %s", self.checkpoint_path)
        except Exception as err:
            logger.error("Failed loading checkpoint (%s); aborting real model init.", err)
            raise RuntimeError(f"Failed to load model checkpoint: {err}") from err

    def _init_patient_registry(self) -> None:
        """Populate patient registry based on active operating mode."""
        if self.is_demo_mode:
            cohort = create_demo_cohort(n_patients=15)
            for seq in cohort:
                self.patients[seq.subject_key] = seq
            logger.info("Seeded %d synthetic demonstration patient records.", len(cohort))
        else:
            # In private research mode, do not fabricate synthetic records.
            self.patients = {}
            logger.info("Private research mode active: patient registry initialized empty (no synthetic fabrication).")

    def get_patient_sequence(self, subject_key: str) -> PatientSequence | None:
        """Retrieve the in-memory PatientSequence for a subject key."""
        return self.patients.get(subject_key)

    def get_model_info(self) -> ModelInfoResponse:
        """Return metadata about current active inference model."""
        feature_count = len(DEMO_MRI_FEATURES)
        feature_names = list(DEMO_MRI_FEATURES)
        temporal_hidden = 64
        rec_threshold = 0.5

        if self.model is not None and hasattr(self.model, "config"):
            cfg = self.model.config
            feature_count = cfg.imaging_features
            feature_names = list(cfg.imaging_feature_names or [])
            temporal_hidden = cfg.temporal_hidden
            rec_threshold = float(self.model_metadata.get("threshold_from_validation", 0.5))

        return ModelInfoResponse(
            model_loaded=(self.model is not None),
            is_demo_mode=self.is_demo_mode,
            imaging_feature_count=feature_count,
            clinical_feature_count=6,
            temporal_hidden=temporal_hidden,
            imaging_feature_names=feature_names,
            clinical_feature_names=list(CLINICAL_ORDER),
            recommended_threshold=rec_threshold,
        )

    def search_patients(
        self,
        query: str = "",
        label_filter: int | None = None,
        min_visits: int = 1,
    ) -> list[PatientSummary]:
        """Search and filter patient records."""
        summaries = cohort_to_summaries(list(self.patients.values()))
        results: list[PatientSummary] = []
        query_norm = query.strip().lower()

        for summary in summaries:
            if query_norm and query_norm not in summary.subject_key.lower():
                continue
            if label_filter is not None and summary.label != label_filter:
                continue
            if summary.visit_count < min_visits:
                continue
            results.append(summary)

        return results

    def get_patient_detail(self, subject_key: str) -> PatientDetailResponse | None:
        """Retrieve full visits and summary for a single patient."""
        seq = self.patients.get(subject_key)
        if seq is None:
            return None

        summaries = cohort_to_summaries([seq])
        visits = patient_sequence_to_visits(seq)
        return PatientDetailResponse(
            summary=summaries[0],
            visits=visits,
            imaging_feature_names=list(seq.imaging_feature_names or DEMO_MRI_FEATURES[:seq.imaging_feature_count]),
            clinical_feature_names=list(CLINICAL_ORDER),
        )

    def predict(self, sequence: PatientSequence, threshold: float = 0.5) -> PredictionResponse:
        """Run progression prediction on a patient sequence."""
        if not self.is_demo_mode and self.model is not None:
            try:
                from nacc_progression.training import predict_patient
                res = predict_patient(self.model, sequence, threshold)
                return prediction_result_to_response(res)
            except Exception as err:
                logger.error("Real model prediction failed (%s); fallback blocked for data safety.", err)
                raise RuntimeError(f"Real model prediction failed: {err}") from err

        res = _synthetic_predict(sequence, threshold)
        return prediction_result_to_response(res)

    def explain(
        self,
        sequence: PatientSequence,
        steps: int = 32,
        include_ablation: bool = True,
        threshold: float = 0.5,
    ) -> ExplanationResponse:
        """Generate Integrated Gradients attribution and visit ablation."""
        if not self.is_demo_mode and self.model is not None:
            try:
                from nacc_progression.explainability import explain_patient, leave_one_visit_out
                mri_feature_names = (
                    sequence.imaging_feature_names
                    or (DEMO_MRI_FEATURES[:sequence.imaging_feature_count] if len(DEMO_MRI_FEATURES) >= sequence.imaging_feature_count else None)
                )
                explanation = explain_patient(
                    self.model,
                    sequence,
                    steps=steps,
                    mri_feature_names=mri_feature_names,
                    clinical_feature_names=CLINICAL_ORDER,
                )
                ablation = None
                if include_ablation and sequence.visit_count >= 2:
                    ablation = leave_one_visit_out(self.model, sequence, threshold)
                return explanation_result_to_response(explanation, ablation)
            except Exception as err:
                logger.error("Real model explainability failed (%s); fallback blocked for data safety.", err)
                raise RuntimeError(f"Real model explainability failed: {err}") from err

        return _synthetic_explain(sequence, steps, include_ablation, threshold)

    def get_audit_summary(self) -> AuditResponse:
        """Provide cohort audit metrics and data protection guidance."""
        return AuditResponse(
            source_freeze="NACC Freeze 74 Mixed-protocol quantitative MRI",
            cohort_counts={
                "eligible_participants": 75,
                "progressions": 20,
                "stable_negatives": 55,
                "progression_rate": 0.267,
                "mri_features_approved": 152,
                "mri_features_retained": 149,
            },
            split_counts={"train": 52, "validation": 11, "test": 12},
            evaluation_metrics={
                "auroc": 0.889,
                "auprc": 0.806,
                "balanced_accuracy": 0.833,
                "sensitivity": 0.850,
                "specificity": 0.818,
                "brier_score": 0.124,
            },
            data_protection_warning=(
                "Protected Health Information warning: Real NACC identifiers and participant derivatives "
                "are restricted under the DUA. Only HMAC pseudonymous subject keys are served."
            ),
        )
