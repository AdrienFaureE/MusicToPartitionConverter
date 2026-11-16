"""Tests de la file de jobs, du pipeline (moteur factice) et de la CLI."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from partition import cli
from partition.config import Settings
from partition.jobs import DONE, ERROR, Job, JobManager
from partition.logging_utils import close_job_logger, get_job_logger, setup_logging
from partition.pipeline import PipelineError, PipelineResult, run_pipeline


def wait_for(manager: JobManager, job_id: str, timeout: float = 60.0) -> Job:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get(job_id)
        assert job is not None
        if job.status in (DONE, ERROR):
            return job
        time.sleep(0.02)
    raise AssertionError("Le job n'a pas terminé à temps.")


def fake_runner(**kwargs: Any) -> PipelineResult:
    output_dir: Path = kwargs["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    kwargs["progress"]("transcription", 0.5, "Transcription")
    if kwargs["engine"] == "mt3":
        raise PipelineError("Moteur mt3 non disponible.")
    (output_dir / "score.musicxml").write_text("<score-partwise/>", encoding="utf-8")
    return PipelineResult(output_dir, kwargs["engine"], {"status": "done"}, {"score.musicxml": output_dir / "score.musicxml"})


# --- file de jobs ----------------------------------------------------------
def test_single_engine_job(settings: Settings, tmp_path: Path) -> None:
    manager = JobManager(settings, runner=fake_runner)
    job = manager.submit(tmp_path / "morceau.mp3", engine="basic_pitch", filename="morceau.mp3")
    done = wait_for(manager, job.id)
    assert done.status == DONE and done.progress == 1.0 and done.finished_at
    assert done.runs[0].files == ["score.musicxml"]
    assert (manager.job_dir(job.id) / "score.musicxml").is_file()
    data = done.to_dict()
    assert data["filename"] == "morceau.mp3" and data["runs"][0]["report"] == {"status": "done"}
    manager.shutdown()


def test_all_engines_run_in_sequence_and_tolerate_failures(settings: Settings, tmp_path: Path) -> None:
    manager = JobManager(settings, runner=fake_runner)
    job = manager.submit(tmp_path / "a.wav", engine="all")
    done = wait_for(manager, job.id)
    by_engine = {run.engine: run for run in done.runs}
    assert list(by_engine) == ["bytedance", "mt3", "basic_pitch"]
    assert done.status == DONE  # au moins un moteur a réussi
    assert by_engine["mt3"].status == ERROR and "non disponible" in (by_engine["mt3"].error or "")
    assert by_engine["bytedance"].files == ["bytedance/score.musicxml"]
    assert (manager.job_dir(job.id) / "basic_pitch" / "score.musicxml").is_file()
    manager.shutdown()


def test_job_fails_when_every_engine_fails(settings: Settings, tmp_path: Path) -> None:
    manager = JobManager(settings, runner=fake_runner)
    done = wait_for(manager, manager.submit(tmp_path / "a.wav", engine="mt3").id)
    assert done.status == ERROR and "mt3" in (done.error or "")
    manager.shutdown()


def test_unknown_engine_is_rejected(settings: Settings, tmp_path: Path) -> None:
    manager = JobManager(settings, runner=fake_runner)
    with pytest.raises(ValueError, match="Moteur inconnu"):
        manager.submit(tmp_path / "a.wav", engine="inexistant")
    assert manager.resolve_engines(None) == [settings.engine]
    manager.shutdown()


def test_progress_is_visible_while_running(settings: Settings, tmp_path: Path) -> None:
    release = threading.Event()

    def slow_runner(**kwargs: Any) -> PipelineResult:
        kwargs["progress"]("transcription", 0.4, "Transcription")
        release.wait(10)
        return fake_runner(**kwargs)

    manager = JobManager(settings, runner=slow_runner)
    job = manager.submit(tmp_path / "a.wav", engine="basic_pitch")
    deadline = time.time() + 10
    while job.progress < 0.4 and time.time() < deadline:
        time.sleep(0.01)
    snapshot = job.to_dict()
    assert snapshot["status"] == "running" and snapshot["progress"] == pytest.approx(0.4)
    assert "Transcription" in snapshot["step"]
    release.set()
    assert wait_for(manager, job.id).status == DONE
    manager.shutdown()


def test_jobs_are_listed_and_reloaded_from_disk(settings: Settings, tmp_path: Path) -> None:
    manager = JobManager(settings, runner=fake_runner)
    job = wait_for(manager, manager.submit(tmp_path / "a.wav", engine="basic_pitch").id)
    assert [j.id for j in manager.list_jobs()] == [job.id]
    manager.shutdown()

    restarted = JobManager(settings, runner=fake_runner)
    reloaded = restarted.get(job.id)
    assert reloaded is not None and reloaded.status == DONE and reloaded.runs[0].files == ["score.musicxml"]
    assert restarted.get("inconnu") is None
    assert restarted.get("../ailleurs") is None
    restarted.shutdown()


# --- pipeline avec moteur factice -----------------------------------------
def test_pipeline_end_to_end_with_fake_engine(
    settings: Settings, fake_engine: str, fixture_files: dict[str, Path], tmp_path: Path
) -> None:
    steps: list[str] = []
    out = tmp_path / "job"
    result = run_pipeline(
        out, engine=fake_engine, audio_path=fixture_files["piano_wav"], settings=settings,
        progress=lambda step, fraction, message: steps.append(step), title="Essai",
    )  # fmt: skip
    for name in ("audio.wav", "raw.mid", "cleaned.mid", "quantized.mid", "score.musicxml", "score.html",
                 "score.pdf", "report.json", "job.log"):  # fmt: skip
        assert (out / name).is_file(), name
        assert name in result.files
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "done" and report["engine"] == "fake"
    assert report["tempo"]["bpm"] == pytest.approx(120, rel=0.03)
    assert report["notes"]["right"] == 16 and report["notes"]["left"] == 4
    assert report["time_signature"] == "4/4" and report["measures"] == 4
    assert report["title"] == "Essai" and report["pdf_method"] == "verovio"
    assert steps[0] == "audio" and steps[-1] == "exports"
    assert "Transcription" in (out / "job.log").read_text(encoding="utf-8")


def test_pipeline_resume_from_raw_and_quantized(
    settings: Settings, fake_engine: str, fixture_files: dict[str, Path], tmp_path: Path
) -> None:
    out = tmp_path / "job"
    first = run_pipeline(out, engine=fake_engine, audio_path=fixture_files["piano_wav"], settings=settings)
    (out / "score.musicxml").unlink()

    settings.hands.split_pitch = 74
    from_raw = run_pipeline(out, settings=settings, start_from="raw")
    assert from_raw.engine == "fake"  # relu dans le rapport précédent
    assert (out / "score.musicxml").is_file()
    assert from_raw.report["notes"]["left"] > first.report["notes"]["left"]
    assert "transcription" not in from_raw.report["timings_s"]

    (out / "hand_overrides.json").write_text(json.dumps({"split_pitch": 0}), encoding="utf-8")
    from_quantized = run_pipeline(out, settings=settings, start_from="quantized")
    assert from_quantized.report["notes"]["left"] == 0 and from_quantized.report["hand_overrides"] is True
    assert from_quantized.report["tempo"]["bpm"] == pytest.approx(first.report["tempo"]["bpm"], abs=0.1)
    assert "cleanup" not in from_quantized.report["timings_s"]


def test_pipeline_errors_are_readable(settings: Settings, fake_engine: str, tmp_path: Path) -> None:
    with pytest.raises(PipelineError, match="introuvable"):
        run_pipeline(tmp_path / "a", engine=fake_engine, audio_path=tmp_path / "absent.wav", settings=settings)
    report = json.loads((tmp_path / "a" / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "error" and "introuvable" in report["error"]
    with pytest.raises(PipelineError, match="Reprise impossible"):
        run_pipeline(tmp_path / "b", settings=settings, start_from="raw")
    with pytest.raises(PipelineError, match="Reprise impossible"):
        run_pipeline(tmp_path / "b", settings=settings, start_from="quantized")
    with pytest.raises(PipelineError, match="Point de reprise inconnu"):
        run_pipeline(tmp_path / "c", settings=settings, start_from="ailleurs")
    with pytest.raises(PipelineError, match="Aucun fichier audio"):
        run_pipeline(tmp_path / "d", engine=fake_engine, settings=settings)


def test_pipeline_unknown_engine_and_bad_override(
    settings: Settings, fake_engine: str, fixture_files: dict[str, Path], tmp_path: Path
) -> None:
    with pytest.raises(PipelineError, match="Moteur inconnu"):
        run_pipeline(tmp_path / "a", engine="inexistant", audio_path=fixture_files["piano_wav"], settings=settings)
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(PipelineError, match="objet JSON"):
        run_pipeline(
            tmp_path / "b", engine=fake_engine, audio_path=fixture_files["piano_wav"],
            settings=settings, overrides_path=bad,
        )  # fmt: skip


# --- journalisation ----------------------------------------------------------
def test_job_logger_writes_to_job_log(tmp_path: Path) -> None:
    assert setup_logging() is setup_logging()
    logger = get_job_logger(tmp_path / "job", "abc")
    assert get_job_logger(tmp_path / "job", "abc") is logger
    assert len([h for h in logger.handlers if hasattr(h, "baseFilename")]) == 1
    logger.info("message de test")
    close_job_logger(logger)
    assert "message de test" in (tmp_path / "job" / "job.log").read_text(encoding="utf-8")
    assert not logger.handlers


# --- CLI -----------------------------------------------------------------------
def test_cli_engines(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["engines"]) == 0
    output = capsys.readouterr().out
    assert "bytedance" in output and "mt3" in output and "basic_pitch" in output


def test_cli_transcribe_and_rebuild(
    fake_engine: str, fixture_files: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    monkeypatch.setenv("PARTITION_TEMPO__METHOD", "midi")
    out = tmp_path / "sortie"
    code = cli.main(["transcribe", str(fixture_files["piano_wav"]), "--engine", fake_engine, "--output", str(out)])
    assert code == 0 and (out / "score.musicxml").is_file()
    assert "Terminé" in capsys.readouterr().out
    assert cli.main(["rebuild", str(out), "--from", "quantized"]) == 0
    assert cli.main(["rebuild", str(tmp_path / "vide")]) == 1
    assert "Erreur" in capsys.readouterr().err


def test_cli_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_engine: str) -> None:
    assert cli.main(["transcribe", str(tmp_path / "absent.mp3"), "--engine", fake_engine, "--output", str(tmp_path / "o")]) == 1
    assert "introuvable" in capsys.readouterr().err
    assert cli.main(["--config", str(tmp_path / "absent.yaml"), "engines"]) == 2
    assert cli.build_parser().parse_args(["serve", "--port", "9000"]).port == 9000
