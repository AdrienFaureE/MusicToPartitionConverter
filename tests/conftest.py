"""Fixtures pytest partagées."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures_generator import generate_all  # noqa: E402

from partition.config import Settings, load_settings  # noqa: E402
from partition.transcription import (  # noqa: E402
    NoteEvent,
    TranscriptionModel,
    register_engine,
    unregister_engine,
)

FAKE_NOTES = [
    # (début s, fin s, hauteur) à 120 à la noire : gamme à droite, basse à gauche
    *[(0.5 * i, 0.5 * i + 0.45, 72 + (i % 5)) for i in range(16)],
    *[(2.0 * i, 2.0 * i + 1.9, 48 - (i % 2) * 5) for i in range(4)],
]


class FakeEngine(TranscriptionModel):
    """Moteur factice : retourne toujours les mêmes notes, sans modèle."""

    name = "fake"
    supports_multi_instrument = False

    def load(self) -> None:
        self._loaded = True

    def transcribe(self, audio_path: Path) -> list[NoteEvent]:
        return [NoteEvent(onset, offset, pitch, 80) for onset, offset, pitch in FAKE_NOTES]


@pytest.fixture(scope="session")
def fixture_files() -> dict[str, Path]:
    """Fichiers MIDI et audio synthétiques (générés au besoin)."""
    return generate_all()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Configuration isolée : données dans un dossier temporaire, aucune variable d'environnement."""
    return load_settings(
        environ={},
        overrides={
            "paths": {"data_dir": str(tmp_path / "data"), "musescore_path": None, "lilypond_path": None},
            "tempo": {"method": "midi"},
        },
    )


@pytest.fixture
def fake_engine() -> Iterator[str]:
    """Enregistre le moteur factice le temps d'un test."""
    register_engine("fake", lambda options, models_dir: FakeEngine(options, models_dir))
    yield "fake"
    unregister_engine("fake")
