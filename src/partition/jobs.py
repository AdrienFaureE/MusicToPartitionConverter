"""File de jobs asynchrones (threads) : état, progression et résultats.

Un job correspond à un fichier audio et à un ou plusieurs moteurs. Avec un seul moteur, les
fichiers sont écrits dans ``data/outputs/<job_id>/`` ; avec plusieurs (``all``), chaque moteur a
son sous-dossier ``data/outputs/<job_id>/<moteur>/``. Les jobs s'exécutent un par un par défaut :
les modèles sont gourmands en mémoire.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from partition.config import Settings
from partition.pipeline import PipelineResult, run_pipeline
from partition.transcription.registry import available_engine_names

ALL_ENGINES = "all"
QUEUED, RUNNING, DONE, ERROR = "queued", "running", "done", "error"

# Signature : runner(output_dir=..., engine=..., audio_path=..., settings=..., progress=..., job_id=..., title=...)
Runner = Callable[..., PipelineResult]


@dataclass
class EngineRun:
    """Exécution d'un moteur au sein d'un job."""

    engine: str
    subdir: str = ""
    status: str = QUEUED
    progress: float = 0.0
    step: str = ""
    error: str | None = None
    report: dict[str, Any] | None = None
    files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Représentation sérialisable."""
        return {
            "engine": self.engine,
            "status": self.status,
            "progress": round(self.progress, 3),
            "step": self.step,
            "error": self.error,
            "report": self.report,
            "files": list(self.files),
        }


@dataclass
class Job:
    """État d'un job."""

    id: str
    filename: str
    engine: str
    runs: list[EngineRun]
    status: str = QUEUED
    created_at: str = ""
    finished_at: str | None = None
    error: str | None = None

    @property
    def progress(self) -> float:
        """Progression globale entre 0 et 1."""
        return sum(run.progress for run in self.runs) / len(self.runs) if self.runs else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Représentation sérialisable."""
        current = next((run for run in self.runs if run.status == RUNNING), None)
        return {
            "id": self.id,
            "filename": self.filename,
            "engine": self.engine,
            "status": self.status,
            "progress": round(self.progress, 3),
            "step": f"{current.engine} : {current.step}" if current else "",
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "runs": [run.to_dict() for run in self.runs],
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobManager:
    """Gère la soumission et le suivi des jobs."""

    def __init__(self, settings: Settings, runner: Runner | None = None, max_workers: int = 1) -> None:
        self.settings = settings
        self._runner: Runner = runner or run_pipeline
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="partition-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    # --- chemins -----------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        """Dossier de sortie d'un job."""
        return self.settings.outputs_dir / job_id

    # --- soumission --------------------------------------------------------
    def resolve_engines(self, engine: str | None) -> list[str]:
        """Liste des moteurs à lancer pour une demande (``all`` = tous les moteurs enregistrés).

        Raises:
            ValueError: nom de moteur inconnu.
        """
        name = engine or self.settings.engine
        if name == ALL_ENGINES:
            return available_engine_names()
        if name not in available_engine_names():
            known = ", ".join([*available_engine_names(), ALL_ENGINES])
            raise ValueError(f"Moteur inconnu : « {name} ». Valeurs acceptées : {known}.")
        return [name]

    def submit(
        self, audio_path: Path, engine: str | None = None, filename: str | None = None, job_id: str | None = None
    ) -> Job:
        """Crée un job et le place dans la file. Retourne immédiatement."""
        engines = self.resolve_engines(engine)
        job_id = job_id or uuid.uuid4().hex[:12]
        multiple = len(engines) > 1
        job = Job(
            id=job_id,
            filename=filename or Path(audio_path).name,
            engine=engine or self.settings.engine,
            runs=[EngineRun(engine=name, subdir=name if multiple else "") for name in engines],
            created_at=_now(),
        )
        self.job_dir(job_id).mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._jobs[job_id] = job
        self._save(job)
        self._executor.submit(self._execute, job, Path(audio_path))
        return job

    # --- consultation ------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        """Retourne un job (en mémoire, sinon relu depuis ``job.json`` après un redémarrage)."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job
        return self._load(job_id)

    def list_jobs(self) -> list[Job]:
        """Jobs connus de ce processus, du plus récent au plus ancien."""
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def shutdown(self, wait: bool = True) -> None:
        """Arrête la file de jobs."""
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    # --- exécution ---------------------------------------------------------
    def _execute(self, job: Job, audio_path: Path) -> None:
        with self._lock:
            job.status = RUNNING
        title = Path(job.filename).stem
        for run in job.runs:
            output_dir = self.job_dir(job.id) / run.subdir if run.subdir else self.job_dir(job.id)

            def on_progress(step: str, fraction: float, message: str, run: EngineRun = run) -> None:
                with self._lock:
                    run.progress, run.step = float(fraction), message

            with self._lock:
                run.status, run.step = RUNNING, "Démarrage"
            try:
                result = self._runner(
                    output_dir=output_dir, engine=run.engine, audio_path=audio_path,
                    settings=self.settings, progress=on_progress, job_id=f"{job.id}-{run.engine}", title=title,
                )  # fmt: skip
                prefix = f"{run.subdir}/" if run.subdir else ""
                with self._lock:
                    run.report = result.report
                    run.files = sorted(prefix + name for name in result.files)
                    run.status, run.progress, run.step = DONE, 1.0, "Terminé"
            except Exception as exc:  # PipelineError porte déjà un message lisible
                with self._lock:
                    run.status, run.progress, run.error, run.step = ERROR, 1.0, str(exc), "Échec"
            self._save(job)
        with self._lock:
            succeeded = [run for run in job.runs if run.status == DONE]
            job.status = DONE if succeeded else ERROR
            if not succeeded:
                job.error = " ; ".join(f"{run.engine} : {run.error}" for run in job.runs)
            job.finished_at = _now()
        self._save(job)

    # --- persistance légère --------------------------------------------------
    def _save(self, job: Job) -> None:
        try:
            with self._lock:
                data = job.to_dict()
                data["subdirs"] = {run.engine: run.subdir for run in job.runs}
            (self.job_dir(job.id) / "job.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass  # la persistance est un confort, pas une nécessité

    def _load(self, job_id: str) -> Job | None:
        path = self.job_dir(job_id) / "job.json"
        if not job_id.isalnum() or not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            subdirs = data.get("subdirs", {})
            runs = [
                EngineRun(
                    engine=item["engine"], subdir=subdirs.get(item["engine"], ""), status=item["status"],
                    progress=item["progress"], step=item["step"], error=item["error"],
                    report=item["report"], files=item["files"],
                )  # fmt: skip
                for item in data["runs"]
            ]
            job = Job(
                id=data["id"], filename=data["filename"], engine=data["engine"], runs=runs,
                status=data["status"], created_at=data["created_at"],
                finished_at=data.get("finished_at"), error=data.get("error"),
            )  # fmt: skip
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            return None
        if job.status in (QUEUED, RUNNING):  # interrompu par un redémarrage du serveur
            job.status, job.error = ERROR, "Job interrompu par un redémarrage du serveur."
        return job
