"""Modèles Pydantic des réponses de l'API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Réponse de ``GET /api/health``."""

    status: str = "ok"
    version: str


class EngineInfo(BaseModel):
    """Statut d'un moteur de transcription."""

    name: str
    available: bool
    supports_multi_instrument: bool
    detail: str = ""


class EnginesResponse(BaseModel):
    """Réponse de ``GET /api/engines``."""

    default: str
    engines: list[EngineInfo]


class FileLink(BaseModel):
    """Fichier téléchargeable d'un job."""

    name: str
    url: str


class EngineRunInfo(BaseModel):
    """Résultat d'un moteur au sein d'un job."""

    engine: str
    status: str
    progress: float = Field(ge=0, le=1)
    step: str = ""
    error: str | None = None
    report: dict[str, Any] | None = None
    files: list[FileLink] = Field(default_factory=list)


class JobResponse(BaseModel):
    """État d'un job."""

    id: str
    filename: str
    engine: str
    status: str
    progress: float = Field(ge=0, le=1)
    step: str = ""
    created_at: str
    finished_at: str | None = None
    error: str | None = None
    runs: list[EngineRunInfo] = Field(default_factory=list)


class JobCreated(BaseModel):
    """Réponse de ``POST /api/jobs``."""

    job_id: str
    status: str
    engines: list[str]


class ErrorResponse(BaseModel):
    """Erreur renvoyée par l'API (message lisible, jamais de traceback)."""

    detail: str
