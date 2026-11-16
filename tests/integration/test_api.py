"""Tests de l'API avec ``TestClient`` et un moteur factice (aucune transcription réelle)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from partition.config import Settings
from partition.jobs import JobManager


@pytest.fixture
def client(settings: Settings, fake_engine: str) -> Iterator[TestClient]:
    settings.engine = fake_engine
    settings.server.max_upload_mb = 1
    app = create_app(settings, JobManager(settings))
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def wait_done(client: TestClient, job_id: str, timeout: float = 120.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get(f"/api/jobs/{job_id}").json()
        if data["status"] in ("done", "error"):
            return data
        time.sleep(0.05)
    raise AssertionError("Le job n'a pas terminé à temps.")


def upload(client: TestClient, path: Path, engine: str | None = "fake", name: str | None = None) -> Any:
    data = {"engine": engine} if engine else {}
    with path.open("rb") as handle:
        return client.post("/api/jobs", files={"file": (name or path.name, handle, "audio/wav")}, data=data)


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok" and response.json()["version"]


def test_engines(client: TestClient) -> None:
    data = client.get("/api/engines").json()
    names = {engine["name"]: engine for engine in data["engines"]}
    assert {"bytedance", "mt3", "basic_pitch", "fake"} <= set(names)
    assert names["fake"]["available"] is True and data["default"] == "fake"
    assert isinstance(names["mt3"]["detail"], str)


def test_full_job_and_downloads(client: TestClient, fixture_files: dict[str, Path]) -> None:
    response = upload(client, fixture_files["piano_wav"], name="Mon morceau.wav")
    assert response.status_code == 202
    created = response.json()
    assert created["engines"] == ["fake"]
    job = wait_done(client, created["job_id"])
    assert job["status"] == "done" and job["progress"] == 1.0 and job["filename"] == "Mon morceau.wav"
    run = job["runs"][0]
    assert run["report"]["title"] == "Mon morceau"
    files = {item["name"]: item["url"] for item in run["files"]}
    assert {"score.musicxml", "score.pdf", "score.html", "raw.mid", "quantized.mid", "report.json"} <= set(files)

    musicxml = client.get(files["score.musicxml"])
    assert musicxml.status_code == 200 and b"<score-partwise" in musicxml.content
    assert "attachment" in musicxml.headers["content-disposition"]
    html = client.get(files["score.html"])
    assert "<svg" in html.text and "inline" in html.headers["content-disposition"]
    assert client.get(files["score.pdf"]).content.startswith(b"%PDF")
    assert client.get(files["raw.mid"]).content.startswith(b"MThd")
    assert client.get(files["report.json"]).json()["engine"] == "fake"
    assert "attachment" in client.get(files["score.html"] + "?download=true").headers["content-disposition"]


def test_default_engine_is_used_when_omitted(client: TestClient, fixture_files: dict[str, Path]) -> None:
    response = upload(client, fixture_files["piano_wav"], engine=None)
    assert response.status_code == 202 and response.json()["engines"] == ["fake"]
    wait_done(client, response.json()["job_id"])


def test_unavailable_engine_gives_clear_error(client: TestClient, fixture_files: dict[str, Path]) -> None:
    response = upload(client, fixture_files["piano_wav"], engine="mt3")
    job = wait_done(client, response.json()["job_id"])
    if job["status"] == "error":  # cas normal : MT3 n'est pas installé
        assert "non disponible" in job["error"]
        assert "Traceback" not in job["error"]


def test_invalid_requests(client: TestClient, fixture_files: dict[str, Path], tmp_path: Path) -> None:
    text = tmp_path / "notes.txt"
    text.write_text("bonjour", encoding="utf-8")
    response = upload(client, text)
    assert response.status_code == 400 and "non pris en charge" in response.json()["detail"]

    response = upload(client, fixture_files["piano_wav"], engine="inexistant")
    assert response.status_code == 400 and "Moteur inconnu" in response.json()["detail"]

    empty = tmp_path / "vide.wav"
    empty.write_bytes(b"")
    assert upload(client, empty).status_code == 400

    big = tmp_path / "gros.wav"
    big.write_bytes(b"0" * (1024 * 1024 + 10))
    response = upload(client, big)
    assert response.status_code == 413 and "volumineux" in response.json()["detail"]

    response = client.post("/api/jobs", data={"engine": "fake"})
    assert response.status_code == 422 and "file" in response.json()["detail"]


def test_corrupted_audio_gives_readable_job_error(client: TestClient, tmp_path: Path) -> None:
    broken = tmp_path / "casse.wav"
    broken.write_bytes(b"pas un wav")
    job = wait_done(client, upload(client, broken).json()["job_id"])
    assert job["status"] == "error"
    assert job["error"] and "Traceback" not in job["error"]


def test_unknown_job_and_file(client: TestClient, fixture_files: dict[str, Path]) -> None:
    assert client.get("/api/jobs/inconnu").status_code == 404
    assert client.get("/api/jobs/inconnu/files/score.pdf").status_code == 404
    job_id = upload(client, fixture_files["piano_wav"]).json()["job_id"]
    wait_done(client, job_id)
    assert client.get(f"/api/jobs/{job_id}/files/absent.pdf").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/files/..%2F..%2Fconfig%2Fdefault.yaml").status_code == 404


def test_static_interface_is_served(client: TestClient) -> None:
    index = client.get("/")
    assert index.status_code == 200 and "partition-ia" in index.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200
