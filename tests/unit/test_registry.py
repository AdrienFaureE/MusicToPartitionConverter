"""Tests du registre des moteurs et de l'interface commune."""

from __future__ import annotations

from pathlib import Path

import pytest

from partition.config import Settings
from partition.transcription import (
    EngineUnavailableError,
    NoteEvent,
    TranscriptionModel,
    available_engine_names,
    engine_statuses,
    get_engine,
    normalize_events,
    register_engine,
    unregister_engine,
)
from partition.transcription.basic_pitch_model import BasicPitchModel
from partition.transcription.bytedance_model import CHECKPOINT_NAME, ByteDanceModel
from partition.transcription.mt3_model import MT3Model, program_to_instrument


def test_builtin_engines_are_registered() -> None:
    assert available_engine_names()[:3] == ["bytedance", "mt3", "basic_pitch"]


def test_get_engine_returns_cached_instance(settings: Settings) -> None:
    first = get_engine("basic_pitch", settings)
    assert isinstance(first, BasicPitchModel)
    assert get_engine("basic_pitch", settings) is first
    assert isinstance(get_engine("bytedance", settings), ByteDanceModel)
    assert isinstance(get_engine("mt3", settings), MT3Model)


def test_unknown_engine(settings: Settings) -> None:
    with pytest.raises(EngineUnavailableError, match="Moteur inconnu"):
        get_engine("inexistant", settings)


def test_register_and_unregister(settings: Settings, fake_engine: str) -> None:
    assert fake_engine in available_engine_names()
    engine = get_engine(fake_engine, settings)
    notes = engine.transcribe(Path("peu_importe.wav"))
    assert notes and all(isinstance(n, NoteEvent) for n in notes)
    unregister_engine(fake_engine)
    assert fake_engine not in available_engine_names()


def test_statuses_never_raise(settings: Settings) -> None:
    class Broken(TranscriptionModel):
        name = "broken"

        def load(self) -> None: ...

        def transcribe(self, audio_path: Path) -> list[NoteEvent]:
            return []

        def check_available(self) -> str | None:
            raise RuntimeError("boum")

    register_engine("broken", lambda options, models_dir: Broken(options, models_dir))
    try:
        statuses = {status.name: status for status in engine_statuses(settings)}
    finally:
        unregister_engine("broken")
    assert set(statuses) >= {"bytedance", "mt3", "basic_pitch", "broken"}
    assert statuses["broken"].available is False and "boum" in statuses["broken"].detail
    assert statuses["mt3"].supports_multi_instrument is True
    assert statuses["bytedance"].supports_multi_instrument is False


def test_bytedance_reports_missing_checkpoint(tmp_path: Path) -> None:
    engine = ByteDanceModel({}, tmp_path)
    assert engine.checkpoint_path == tmp_path / "bytedance" / CHECKPOINT_NAME
    reason = engine.check_available()
    assert reason is not None
    with pytest.raises(EngineUnavailableError):
        engine.load()


def test_mt3_unavailable_does_not_break(tmp_path: Path) -> None:
    engine = MT3Model({}, tmp_path)
    assert engine.checkpoint_dir == tmp_path / "mt3" / "checkpoints" / "mt3"
    status = engine.status()
    assert status.available is False and status.detail
    with pytest.raises(EngineUnavailableError):
        engine.load()
    assert MT3Model({"model_type": "inconnu"}, tmp_path).check_available() is not None


def test_program_to_instrument() -> None:
    assert program_to_instrument(0, False) == "piano"
    assert program_to_instrument(4, False) == "piano"
    assert program_to_instrument(40, False) == "violin"
    assert program_to_instrument(0, True) == "drums"


def test_normalize_events() -> None:
    events = [
        NoteEvent(2.0, 3.0, 64, 300),
        NoteEvent(-0.5, 1.0, 60, 0),
        NoteEvent(1.0, 1.0, 62, 80),  # durée nulle : écartée
        NoteEvent(0.0, 1.0, 200, 80),  # hauteur hors limites : écartée
    ]
    result = normalize_events(events)
    assert [(n.onset, n.pitch, n.velocity) for n in result] == [(0.0, 60, 1), (2.0, 64, 127)]


def test_note_event_dict_round_trip() -> None:
    note = NoteEvent(1.23456, 2.5, 61, 77, instrument="flute", confidence=0.87654)
    restored = NoteEvent.from_dict(note.to_dict())
    assert restored.pitch == 61 and restored.instrument == "flute"
    assert restored.onset == pytest.approx(1.2346) and restored.confidence == pytest.approx(0.8765)
    assert note.duration == pytest.approx(1.26544)
