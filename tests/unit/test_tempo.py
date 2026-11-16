"""Tests de l'estimation du tempo et de la grille."""

from __future__ import annotations

from pathlib import Path

import pytest

from partition.midi_io import midi_to_notes
from partition.postprocess.tempo import (
    BeatGrid,
    constant_grid,
    estimate_grid,
    estimate_grid_from_audio,
    estimate_grid_from_notes,
    estimate_tempo_from_notes,
    fold_bpm,
)
from partition.transcription import NoteEvent


def test_tempo_of_synthetic_midi(fixture_files: dict[str, Path]) -> None:
    notes = midi_to_notes(fixture_files["tempo_90"])
    assert estimate_tempo_from_notes(notes) == pytest.approx(90.0, rel=0.03)


@pytest.mark.parametrize("bpm", [60.0, 100.0, 132.0])
def test_tempo_of_regular_pulse(bpm: float) -> None:
    period = 60.0 / bpm
    notes = [NoteEvent(i * period, i * period + 0.2, 60, 80) for i in range(48)]
    assert estimate_tempo_from_notes(notes) == pytest.approx(bpm, rel=0.03)


def test_tempo_default_when_too_few_notes() -> None:
    assert estimate_tempo_from_notes([NoteEvent(0, 1, 60, 80)], default_bpm=120.0) == 120.0


def test_fold_bpm() -> None:
    assert fold_bpm(45, 50, 180) == 90
    assert fold_bpm(240, 50, 180) == 120
    assert fold_bpm(100, 50, 180) == 100
    with pytest.raises(ValueError):
        fold_bpm(0, 50, 180)


def test_constant_grid_conversions() -> None:
    grid = constant_grid(120.0, 1.0, 5.0)
    assert grid.time_to_beat(1.0) == pytest.approx(0.0)
    assert grid.time_to_beat(2.25) == pytest.approx(2.5)
    assert grid.time_to_beat(0.5) == pytest.approx(-1.0)  # extrapolation avant
    assert grid.time_to_beat(100.0) == pytest.approx(198.0)  # extrapolation après
    assert grid.beat_to_time(2.5) == pytest.approx(2.25)
    assert grid.beat_to_time(-1.0) == pytest.approx(0.5)
    assert grid.beat_to_time(198.0) == pytest.approx(100.0)


def test_variable_grid_follows_beats() -> None:
    grid = BeatGrid(bpm=100, beat_times=[0.0, 0.5, 1.5, 2.0], method="audio-beats")
    assert grid.time_to_beat(1.0) == pytest.approx(1.5)  # milieu d'un battement étiré
    assert grid.beat_to_time(1.5) == pytest.approx(1.0)


def test_beat_grid_with_single_time_is_completed() -> None:
    assert BeatGrid(bpm=60, beat_times=[2.0]).beat_times == [2.0, 3.0]


def test_grid_from_notes_starts_on_first_onset(fixture_files: dict[str, Path]) -> None:
    notes = [NoteEvent(n.onset + 0.8, n.offset + 0.8, n.pitch, n.velocity) for n in midi_to_notes(fixture_files["tempo_90"])]
    grid = estimate_grid_from_notes(notes)
    assert grid.method == "midi"
    assert grid.beat_times[0] == pytest.approx(0.8, abs=0.03)
    assert grid.beat_times[-1] >= max(n.offset for n in notes)


def test_forced_bpm() -> None:
    notes = [NoteEvent(i * 0.5, i * 0.5 + 0.2, 60, 80) for i in range(16)]
    grid, warnings = estimate_grid(notes, forced_bpm=77)
    assert grid.bpm == 77 and grid.method == "forced" and warnings == []


def test_estimate_grid_falls_back_to_notes_without_audio() -> None:
    notes = [NoteEvent(i * 0.5, i * 0.5 + 0.2, 60, 80) for i in range(32)]
    grid, warnings = estimate_grid(notes, audio_path=None, method="audio")
    assert grid.method == "midi"
    assert grid.bpm == pytest.approx(120, rel=0.03)
    assert warnings


def test_grid_from_notes_without_notes() -> None:
    assert estimate_grid_from_notes([]).bpm == 120.0


def test_grid_from_audio(fixture_files: dict[str, Path]) -> None:
    grid = estimate_grid_from_audio(fixture_files["piano_wav"], follow_beats=True)
    assert grid.method == "audio-beats"
    assert len(grid.beat_times) >= 4
    # Le suivi de battements peut se caler à l'octave (60 ou 240) : on tolère ce rapport.
    assert min(abs(grid.bpm / 120 - ratio) for ratio in (0.5, 1.0, 2.0)) < 0.08
    constant = estimate_grid_from_audio(fixture_files["piano_wav"], follow_beats=False)
    assert constant.method == "audio-constant"
