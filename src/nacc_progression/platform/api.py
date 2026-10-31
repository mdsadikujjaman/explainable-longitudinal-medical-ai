"""FastAPI application exposing prediction, explainability, auth, and audit endpoints."""

from __future__ import annotations

import logging
import time
from typing import Any
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from nacc_progression.platform.database import (
    create_user,
    get_prediction_logs,
    get_user_by_username,
    init_db,
    log_prediction,
    update_user_password,
)
from nacc_progression.platform.schemas import (
    AuditResponse,
    ExplanationRequest,
    ExplanationResponse,
    LoginRequest,
    ModelInfoResponse,
    PatientDetailResponse,
    PatientSummary,
    PredictionRequest,
    PredictionResponse,
    TokenResponse,
    UserInfo,
    request_to_patient_sequence,
)
from nacc_progression.platform.security import (
    create_access_token,
    get_current_user,
    hash_password,
    needs_rehash,
    verify_password,
)
from nacc_progression.platform.services import PlatformService

logger = logging.getLogger(__name__)


def create_app(service: PlatformService | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    init_db()

    app = FastAPI(
        title="Explainable Longitudinal Medical AI Platform",
        description="FastAPI Backend for Alzheimer's Disease Progression Prediction and Multimodal XAI.",
        version="0.1.0",
    )

    # Enable CORS for Streamlit frontend and local dev environments
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if service is None:
        service = PlatformService.get_instance()

    # ========================================================================
    # System & Health
    # ========================================================================

    @app.get("/health", tags=["System"])
    @app.get("/api/health", tags=["System"])
    @app.get("/api/v1/health", tags=["System"])
    def health_check() -> dict[str, Any]:
        """Health check endpoint for container orchestrators and monitoring."""
        return {
            "status": "healthy",
            "version": "0.1.0",
            "demo_mode": service.is_demo_mode,
            "timestamp": time.time(),
        }

    # ========================================================================
    # Authentication
    # ========================================================================

    @app.post("/api/auth/login", response_model=TokenResponse, tags=["Auth"])
    @app.post("/api/v1/auth/login", response_model=TokenResponse, tags=["Auth"])
    def login(payload: LoginRequest) -> TokenResponse:
        """Authenticate user credentials and issue an access token."""
        user = get_user_by_username(payload.username)
        if user is None or not verify_password(payload.password, user["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Automatic transparent migration: upgrade legacy Bcrypt or outdated hashes to Argon2id
        if needs_rehash(user["hashed_password"]):
            new_hash = hash_password(payload.password)
            update_user_password(user["username"], new_hash)

        token = create_access_token(subject=user["username"], role=user["role"])
        return TokenResponse(
            access_token=token,
            username=user["username"],
            role=user["role"],
        )

    @app.post("/api/auth/register", response_model=UserInfo, tags=["Auth"])
    @app.post("/api/v1/auth/register", response_model=UserInfo, tags=["Auth"])
    def register(payload: LoginRequest, role: str = "researcher") -> UserInfo:
        """Register a new user account."""
        try:
            created = create_user(payload.username, payload.password, role=role)
            return UserInfo(username=created["username"], role=created["role"])
        except ValueError as err:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))

    @app.get("/api/auth/me", response_model=UserInfo, tags=["Auth"])
    @app.get("/api/v1/auth/me", response_model=UserInfo, tags=["Auth"])
    def get_me(current_user: dict[str, Any] = Depends(get_current_user)) -> UserInfo:
        """Return the current authenticated user's profile."""
        return UserInfo(username=current_user["username"], role=current_user["role"])

    # ========================================================================
    # Model Metadata
    # ========================================================================

    @app.get("/api/model/info", response_model=ModelInfoResponse, tags=["Model"])
    @app.get("/api/v1/model/info", response_model=ModelInfoResponse, tags=["Model"])
    def get_model_info() -> ModelInfoResponse:
        """Get model status, parameters, and input feature specifications."""
        return service.get_model_info()

    # ========================================================================
    # Patient Directory & Records
    # ========================================================================

    @app.get("/api/patients", response_model=list[PatientSummary], tags=["Patients"])
    @app.get("/api/v1/patients", response_model=list[PatientSummary], tags=["Patients"])
    def list_patients(
        query: str = Query("", description="Filter by subject key substring"),
        label: int | None = Query(None, description="Filter by outcome label (0: stable, 1: progressor)"),
        min_visits: int = Query(1, ge=1, le=3, description="Minimum sequence visits"),
    ) -> list[PatientSummary]:
        """Search and filter patient records."""
        return service.search_patients(query=query, label_filter=label, min_visits=min_visits)

    @app.get("/api/patients/{subject_key}", response_model=PatientDetailResponse, tags=["Patients"])
    @app.get("/api/v1/patients/{subject_key}", response_model=PatientDetailResponse, tags=["Patients"])
    def get_patient(subject_key: str) -> PatientDetailResponse:
        """Fetch preprocessed longitudinal visits and summary for a single patient."""
        detail = service.get_patient_detail(subject_key)
        if detail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient '{subject_key}' not found",
            )
        return detail

    # ========================================================================
    # Prediction
    # ========================================================================

    @app.post("/api/predict", response_model=PredictionResponse, tags=["Inference"])
    @app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Inference"])
    def predict_progression(
        payload: PredictionRequest,
        current_user: dict[str, Any] | None = Depends(get_current_user),
    ) -> PredictionResponse:
        """Execute multimodal prediction on a patient visit sequence."""
        sequence = request_to_patient_sequence(payload)
        try:
            response = service.predict(sequence, threshold=payload.threshold)
        except RuntimeError:
            logger.exception("Prediction calculation failed for subject: %s", sequence.subject_key)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction processing failed. An internal error occurred during model evaluation.",
            )

        # Audit logging
        username = current_user["username"] if current_user else "anonymous"
        log_prediction(
            username=username,
            subject_key=response.subject_key,
            probability=response.probability,
            threshold=response.threshold,
            predicted_label=response.predicted_label,
            visit_count=response.visit_count,
        )

        return response

    @app.post(
        "/api/v1/patients/{subject_key}/predict",
        response_model=PredictionResponse,
        tags=["Inference"],
    )
    def predict_patient_by_id(
        subject_key: str,
        threshold: float = Query(0.5, ge=0.0, le=1.0, description="Decision threshold"),
        current_user: dict[str, Any] | None = Depends(get_current_user),
    ) -> PredictionResponse:
        """Execute multimodal prediction on a patient record retrieved from the registry."""
        if not service.is_demo_mode and len(service.patients) == 0:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Patient registry is unavailable in private research mode. Real patient records have not been ingested.",
            )

        sequence = service.get_patient_sequence(subject_key)
        if sequence is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient '{subject_key}' not found in registry",
            )

        try:
            response = service.predict(sequence, threshold=threshold)
        except RuntimeError:
            logger.exception("Inference failed for patient: %s", subject_key)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction processing failed. An internal error occurred during model evaluation.",
            )

        username = current_user["username"] if current_user else "anonymous"
        log_prediction(
            username=username,
            subject_key=response.subject_key,
            probability=response.probability,
            threshold=response.threshold,
            predicted_label=response.predicted_label,
            visit_count=response.visit_count,
        )

        return response

    # ========================================================================
    # Explainability (XAI)
    # ========================================================================

    @app.post("/api/explain", response_model=ExplanationResponse, tags=["Explainability"])
    @app.post("/api/v1/explain", response_model=ExplanationResponse, tags=["Explainability"])
    def explain_progression(
        payload: ExplanationRequest,
        current_user: dict[str, Any] | None = Depends(get_current_user),
    ) -> ExplanationResponse:
        """Compute Integrated Gradients feature attributions and visit ablations."""
        sequence = request_to_patient_sequence(payload)
        try:
            return service.explain(
                sequence,
                steps=payload.steps,
                include_ablation=payload.include_visit_ablation,
                threshold=payload.threshold,
            )
        except RuntimeError:
            logger.exception("Explainability calculation failed for subject: %s", sequence.subject_key)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Explainability processing failed. An internal error occurred during attribution calculation.",
            )

    @app.get(
        "/api/v1/patients/{subject_key}/explanation",
        response_model=ExplanationResponse,
        tags=["Explainability"],
    )
    def explain_patient_by_id(
        subject_key: str,
        steps: int = Query(32, ge=2, le=128, description="Integrated Gradients step count"),
        include_visit_ablation: bool = Query(True, description="Include leave-one-visit-out ablation"),
        threshold: float = Query(0.5, ge=0.0, le=1.0, description="Decision threshold"),
        current_user: dict[str, Any] | None = Depends(get_current_user),
    ) -> ExplanationResponse:
        """Compute Integrated Gradients and visit ablation for a patient retrieved from the registry."""
        if not service.is_demo_mode and len(service.patients) == 0:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Patient registry is unavailable in private research mode. Real patient records have not been ingested.",
            )

        sequence = service.get_patient_sequence(subject_key)
        if sequence is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient '{subject_key}' not found in registry",
            )

        try:
            return service.explain(
                sequence,
                steps=steps,
                include_ablation=include_visit_ablation,
                threshold=threshold,
            )
        except RuntimeError:
            logger.exception("Explainability calculation failed for patient: %s", subject_key)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Explainability processing failed. An internal error occurred during attribution calculation.",
            )

    # ========================================================================
    # Dataset Audit & Monitoring
    # ========================================================================

    @app.get("/api/audit", response_model=AuditResponse, tags=["Audit"])
    @app.get("/api/v1/audit", response_model=AuditResponse, tags=["Audit"])
    @app.get("/api/v1/audit/summary", response_model=AuditResponse, tags=["Audit"])
    def get_audit() -> AuditResponse:
        """Retrieve cohort audit metrics, split breakdown, and test performance."""
        return service.get_audit_summary()

    @app.get("/api/audit/logs", tags=["Audit"])
    @app.get("/api/v1/audit/logs", tags=["Audit"])
    def get_audit_logs(
        limit: int = Query(50, ge=1, le=200),
        current_user: dict[str, Any] = Depends(get_current_user),
    ) -> list[dict[str, Any]]:
        """Retrieve recent prediction audit logs."""
        return get_prediction_logs(limit=limit)

    return app


app = create_app()
