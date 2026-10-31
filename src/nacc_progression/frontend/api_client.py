"""Client for FastAPI backend communication with automatic in-process fallback."""

from __future__ import annotations

import os
from typing import Any
import requests

from nacc_progression.platform.schemas import (
    AuditResponse,
    ExplanationRequest,
    ExplanationResponse,
    ModelInfoResponse,
    PatientDetailResponse,
    PatientSummary,
    PredictionRequest,
    PredictionResponse,
    request_to_patient_sequence,
)
from nacc_progression.platform.services import PlatformService

DEFAULT_API_BASE_URL = os.environ.get("BACKEND_API_URL", "http://127.0.0.1:8000")


class APIClient:
    """Unified client providing seamless access to FastAPI or in-memory service."""

    def __init__(self, base_url: str = DEFAULT_API_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self.token: str | None = None
        self.current_user: dict[str, str] | None = None
        self.service = PlatformService.get_instance()
        self._backend_alive: bool | None = None

    def is_backend_online(self) -> bool:
        """Check if standalone FastAPI server is reachable."""
        try:
            r = requests.get(f"{self.base_url}/api/health", timeout=1.5)
            self._backend_alive = (r.status_code == 200)
            return self._backend_alive
        except Exception:
            self._backend_alive = False
            return False

    def login(self, username: str, password: str) -> bool:
        """Authenticate user against API or local database."""
        if self.is_backend_online():
            try:
                res = requests.post(
                    f"{self.base_url}/api/auth/login",
                    json={"username": username, "password": password},
                    timeout=3.0,
                )
                if res.status_code == 200:
                    data = res.json()
                    self.token = data["access_token"]
                    self.current_user = {"username": data["username"], "role": data["role"]}
                    return True
            except Exception:
                pass

        # In-process authentication fallback
        from nacc_progression.platform.database import get_user_by_username, update_user_password
        from nacc_progression.platform.security import (
            create_access_token,
            hash_password,
            needs_rehash,
            verify_password,
        )

        user = get_user_by_username(username)
        if user and verify_password(password, user["hashed_password"]):
            if needs_rehash(user["hashed_password"]):
                update_user_password(user["username"], hash_password(password))
            self.token = create_access_token(user["username"], role=user["role"])
            self.current_user = {"username": user["username"], "role": user["role"]}
            return True

        return False

    def logout(self) -> None:
        """Clear local session credentials."""
        self.token = None
        self.current_user = None

    def get_model_info(self) -> ModelInfoResponse:
        """Fetch model metadata."""
        if self._backend_alive:
            try:
                res = requests.get(f"{self.base_url}/api/model/info", timeout=2.0)
                if res.status_code == 200:
                    return ModelInfoResponse(**res.json())
            except Exception:
                pass
        return self.service.get_model_info()

    def search_patients(
        self,
        query: str = "",
        label: int | None = None,
        min_visits: int = 1,
    ) -> list[PatientSummary]:
        """Query patient records."""
        if self._backend_alive:
            try:
                params: dict[str, Any] = {"query": query, "min_visits": min_visits}
                if label is not None:
                    params["label"] = label
                res = requests.get(f"{self.base_url}/api/patients", params=params, timeout=3.0)
                if res.status_code == 200:
                    return [PatientSummary(**item) for item in res.json()]
            except Exception:
                pass
        return self.service.search_patients(query=query, label_filter=label, min_visits=min_visits)

    def get_patient_detail(self, subject_key: str) -> PatientDetailResponse | None:
        """Retrieve full visits for one patient."""
        if self._backend_alive:
            try:
                res = requests.get(f"{self.base_url}/api/patients/{subject_key}", timeout=3.0)
                if res.status_code == 200:
                    return PatientDetailResponse(**res.json())
            except Exception:
                pass
        return self.service.get_patient_detail(subject_key)

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Trigger progression prediction."""
        if self._backend_alive:
            try:
                headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
                res = requests.post(
                    f"{self.base_url}/api/predict",
                    json=request.model_dump(),
                    headers=headers,
                    timeout=5.0,
                )
                if res.status_code == 200:
                    return PredictionResponse(**res.json())
            except Exception:
                pass

        seq = request_to_patient_sequence(request)
        return self.service.predict(seq, threshold=request.threshold)

    def explain(self, request: ExplanationRequest) -> ExplanationResponse:
        """Request Integrated Gradients XAI attribution."""
        if self._backend_alive:
            try:
                headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
                res = requests.post(
                    f"{self.base_url}/api/explain",
                    json=request.model_dump(),
                    headers=headers,
                    timeout=10.0,
                )
                if res.status_code == 200:
                    return ExplanationResponse(**res.json())
            except Exception:
                pass

        seq = request_to_patient_sequence(request)
        return self.service.explain(
            seq,
            steps=request.steps,
            include_ablation=request.include_visit_ablation,
            threshold=request.threshold,
        )

    def get_audit(self) -> AuditResponse:
        """Fetch dataset audit report."""
        if self._backend_alive:
            try:
                res = requests.get(f"{self.base_url}/api/audit", timeout=2.0)
                if res.status_code == 200:
                    return AuditResponse(**res.json())
            except Exception:
                pass
        return self.service.get_audit_summary()
