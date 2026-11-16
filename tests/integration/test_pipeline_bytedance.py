"""Test d'intégration du moteur ByteDance sur un audio synthétique de dix secondes.

Ignoré si les dépendances ou les poids du moteur sont absents.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from music21 import converter

from partition.config import Settings
from partition.midi_io import midi_to_notes
from partition.pipeline import run_pipeline
from partition.transcription import get_engine

ENGINE = "bytedance"

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_pipeline_bytedance(settings: Settings, fixture_files: dict[str, Path], tmp_path: Path) -> None:
    status = get_engine(ENGINE, settings).status()
    if not status.available:
        pytest.skip(f"Moteur {ENGINE} indisponible : {status.detail}")

    result = run_pipeline(tmp_path / "job", engine=ENGINE, audio_path=fixture_files["piano_wav"], settings=settings)

    raw = result.files["raw.mid"]
    musicxml = result.files["score.musicxml"]
    assert raw.stat().st_size > 0 and musicxml.stat().st_size > 0
    notes = midi_to_notes(raw)
    assert len(notes) >= 10
    assert all(0 <= n.pitch <= 127 and n.offset > n.onset for n in notes)
    parsed = converter.parse(str(musicxml))
    assert len(parsed.parts) == 2
    assert len(parsed.flatten().notes) > 0
    assert result.report["status"] == "done" and result.report["engine"] == ENGINE
    assert result.files["quantized.mid"].stat().st_size > 0
