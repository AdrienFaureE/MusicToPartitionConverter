"""Application FastAPI : API sous ``/api`` et interface web statique sous ``/``."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router
from partition import __version__
from partition.config import Settings, load_settings
from partition.jobs import JobManager
from partition.logging_utils import setup_logging

logger = logging.getLogger("partition.api")


def create_app(settings: Settings | None = None, job_manager: JobManager | None = None) -> FastAPI:
    """Construit l'application (configuration et file de jobs injectables pour les tests)."""
    setup_logging()
    settings = settings or load_settings()
    manager = job_manager or JobManager(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        manager.shutdown(wait=False)

    app = FastAPI(
        title="partition-ia",
        version=__version__,
        description="Transcription locale audio piano vers partition.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.jobs = manager
    app.include_router(router)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = ", ".join(str(error["loc"][-1]) for error in exc.errors())
        return JSONResponse(status_code=422, content={"detail": f"Requête invalide (champs concernés : {fields})."})

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Erreur inattendue sur %s", request.url.path)
        return JSONResponse(
            status_code=500, content={"detail": "Erreur interne du serveur. Consultez les journaux du serveur."}
        )

    web_dir = settings.root / "web"
    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    return app


app = create_app()
