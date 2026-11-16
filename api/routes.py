"""Routes de l'API."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from api.schemas import (
    EngineInfo,
    EngineRunInfo,
    EnginesResponse,
    FileLink,
    HealthResponse,
    JobCreated,
    JobResponse,
)
from partition import __version__
from partition.audio.load import SUPPORTED_EXTENSIONS
from partition.config import Settings
from partition.jobs import Job, JobManager
from partition.transcription.registry import engine_statuses

router = APIRouter(prefix="/api")

_MEDIA_TYPES = {
    ".musicxml": "application/vnd.recordare.musicxml+xml",
    ".pdf": "application/pdf",
    ".mid": "audio/midi",
    ".html": "text/html; charset=utf-8",
    ".json": "application/json",
    ".log": "text/plain; charset=utf-8",
    ".wav": "audio/wav",
}
_CHUNK = 1024 * 1024


def _settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def _manager(request: Request) -> JobManager:
    return request.app.state.jobs  # type: ignore[no-any-return]


def _job_response(job: Job) -> JobResponse:
    data = job.to_dict()
    runs = [
        EngineRunInfo(
            engine=run["engine"], status=run["status"], progress=run["progress"], step=run["step"],
            error=run["error"], report=run["report"],
            files=[FileLink(name=name, url=f"/api/jobs/{job.id}/files/{name}") for name in run["files"]],
        )  # fmt: skip
        for run in data["runs"]
    ]
    return JobResponse(**{**data, "runs": runs})


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Vérifie que le serveur répond."""
    return HealthResponse(version=__version__)


@router.get("/engines", response_model=EnginesResponse)
def engines(request: Request) -> EnginesResponse:
    """Liste les moteurs et leur disponibilité."""
    settings = _settings(request)
    statuses = [EngineInfo(**vars(status)) for status in engine_statuses(settings)]
    return EnginesResponse(default=settings.engine, engines=statuses)


@router.post("/jobs", response_model=JobCreated, status_code=202)
async def create_job(
    request: Request,
    file: UploadFile = File(..., description="Fichier audio (mp3, wav, flac, m4a)."),
    engine: str | None = Form(None, description="bytedance, mt3, basic_pitch ou all."),
) -> JobCreated:
    """Envoie un fichier audio et lance sa transcription."""
    settings, manager = _settings(request), _manager(request)
    filename = Path(file.filename or "audio").name
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format « {extension or '?'} » non pris en charge. Formats acceptés : {', '.join(SUPPORTED_EXTENSIONS)}.",
        )
    try:
        engine_names = manager.resolve_engines(engine)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = uuid.uuid4().hex[:12]
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.uploads_dir / f"{job_id}{extension}"
    limit = settings.server.max_upload_mb * 1024 * 1024
    size = 0
    try:
        with destination.open("wb") as output:
            while chunk := await file.read(_CHUNK):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Fichier trop volumineux (maximum : {settings.server.max_upload_mb} Mo).",
                    )
                output.write(chunk)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    if size == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Le fichier envoyé est vide.")

    job = manager.submit(destination, engine=engine, filename=filename, job_id=job_id)
    return JobCreated(job_id=job.id, status=job.status, engines=engine_names)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, request: Request) -> JobResponse:
    """État et progression d'un job."""
    job = _manager(request).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job introuvable.")
    return _job_response(job)


@router.get("/jobs/{job_id}/files/{name:path}")
def get_file(job_id: str, name: str, request: Request, download: bool = False) -> FileResponse:
    """Télécharge un fichier produit par un job (MusicXML, PDF, MIDI, HTML, rapport)."""
    manager = _manager(request)
    if manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job introuvable.")
    job_dir = manager.job_dir(job_id).resolve()
    target = (job_dir / name).resolve()
    if job_dir not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    media_type = _MEDIA_TYPES.get(target.suffix.lower(), "application/octet-stream")
    inline = target.suffix.lower() in (".html", ".pdf", ".json", ".log") and not download
    return FileResponse(
        target,
        media_type=media_type,
        filename=target.name,
        content_disposition_type="inline" if inline else "attachment",
    )
