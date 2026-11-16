"""Tests de la quantification."""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from partition.postprocess.quantize import grid_step, quantize_notes, snap
from partition.postprocess.tempo import constant_grid
from partition.transcription import NoteEvent

GRID_120 = constant_grid(120.0, 0.0, 20.0)  # une noire = 0,5 s


def test_grid_step() -> None:
    assert grid_step(16) == F(1, 4)
    assert grid_step(8) == F(1, 2)
    with pytest.raises(ValueError):
        grid_step(0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, F(0)), (0.12, F(0)), (0.13, F(1, 4)), (0.99, F(1)), (2.6, F(5, 2)), (2.63, F(11, 4))],
)
def test_snap_sixteenth(value: float, expected: F) -> None:
    assert snap(value, 16) == expected


def test_snap_triplets() -> None:
    assert snap(0.34, 16, triplets=True) == F(1, 3)
    assert snap(0.66, 16, triplets=True) == F(2, 3)
    assert snap(0.26, 16, triplets=True) == F(1, 4)
    assert snap(0.34, 16, triplets=False) == F(1, 4)


def test_onsets_and_durations_are_aligned() -> None:
    notes = [NoteEvent(0.02, 0.49, 60, 80), NoteEvent(0.51, 0.74, 62, 80), NoteEvent(0.76, 1.52, 64, 80)]
    result = quantize_notes(notes, GRID_120, fill_gap_beats=0)
    assert [(n.start, n.duration) for n in result] == [(F(0), F(1)), (F(1), F(1, 2)), (F(3, 2), F(3, 2))]


def test_minimum_duration_is_one_step() -> None:
    result = quantize_notes([NoteEvent(1.0, 1.02, 60, 80)], GRID_120)
    assert result[0].start == F(0) or result[0].start == F(2)
    assert result[0].duration == F(1, 4)


def test_piece_is_shifted_to_first_beat() -> None:
    notes = [NoteEvent(5.0, 5.5, 60, 80), NoteEvent(5.75, 6.0, 62, 80)]
    result = quantize_notes(notes, GRID_120)
    assert result[0].start == F(0)
    assert result[1].start == F(3, 2)


def test_long_rests_are_kept_and_small_gaps_filled() -> None:
    notes = [
        NoteEvent(0.0, 0.4, 60, 80),  # fin à 0,8 noire → 0,75 ; trou de 0,25 avant la suivante : absorbé
        NoteEvent(0.5, 1.0, 62, 80),  # fin à 2 noires ; la suivante est à 6 : silence conservé
        NoteEvent(3.0, 3.5, 64, 80),
    ]
    result = quantize_notes(notes, GRID_120, fill_gap_beats=0.25)
    assert result[0].end == result[1].start == F(1)
    assert result[1].end == F(2)
    assert result[2].start == F(6)


def test_duplicates_on_grid_are_merged() -> None:
    notes = [NoteEvent(0.0, 0.5, 60, 70), NoteEvent(0.03, 1.0, 60, 90)]
    result = quantize_notes(notes, GRID_120)
    assert len(result) == 1
    assert result[0].duration == F(2) and result[0].velocity == 90


def test_restruck_pitch_shortens_previous_note() -> None:
    notes = [NoteEvent(0.0, 2.0, 60, 80), NoteEvent(1.0, 2.0, 60, 80)]
    result = quantize_notes(notes, GRID_120)
    assert [(n.start, n.duration) for n in result] == [(F(0), F(2)), (F(2), F(2))]


def test_instruments_are_preserved() -> None:
    notes = [NoteEvent(0.0, 0.5, 60, 80), NoteEvent(0.0, 0.5, 60, 80, instrument="violin")]
    result = quantize_notes(notes, GRID_120)
    assert sorted(n.instrument for n in result) == ["piano", "violin"]


def test_empty_input() -> None:
    assert quantize_notes([], GRID_120) == []
