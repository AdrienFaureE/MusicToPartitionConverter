"""Tests du nettoyage des notes."""

from __future__ import annotations

from partition.postprocess.cleanup import (
    apply_velocity_threshold,
    clean_notes,
    merge_overlapping,
    remove_short_notes,
)
from partition.transcription import NoteEvent


def test_remove_short_notes() -> None:
    notes = [NoteEvent(0.0, 0.02, 60, 80), NoteEvent(0.0, 0.03, 62, 80), NoteEvent(1.0, 1.5, 64, 80)]
    assert [n.pitch for n in remove_short_notes(notes, 0.03)] == [62, 64]


def test_velocity_threshold() -> None:
    notes = [NoteEvent(0, 1, 60, 5), NoteEvent(0, 1, 62, 12), NoteEvent(0, 1, 64, 100)]
    assert [n.pitch for n in apply_velocity_threshold(notes, 12)] == [62, 64]
    assert len(apply_velocity_threshold(notes, 0)) == 3


def test_overlapping_duplicates_are_merged() -> None:
    notes = [NoteEvent(1.00, 1.50, 60, 70, confidence=0.4), NoteEvent(1.02, 1.80, 60, 90, confidence=0.9)]
    merged = merge_overlapping(notes)
    assert len(merged) == 1
    assert (merged[0].onset, merged[0].offset, merged[0].velocity) == (1.00, 1.80, 90)
    assert merged[0].confidence == 0.9


def test_exact_duplicates_are_merged() -> None:
    notes = [NoteEvent(0.0, 1.0, 60, 80), NoteEvent(0.0, 1.0, 60, 80), NoteEvent(0.0, 1.0, 60, 60)]
    assert len(merge_overlapping(notes)) == 1


def test_restruck_note_cuts_the_previous_one() -> None:
    notes = [NoteEvent(0.0, 2.0, 60, 80), NoteEvent(1.0, 3.0, 60, 80)]
    merged = merge_overlapping(notes)
    assert [(n.onset, n.offset) for n in merged] == [(0.0, 1.0), (1.0, 3.0)]


def test_repeated_notes_are_kept() -> None:
    notes = [NoteEvent(0.0, 0.5, 60, 80), NoteEvent(0.5, 1.0, 60, 80), NoteEvent(1.0, 1.5, 60, 80)]
    assert len(merge_overlapping(notes)) == 3


def test_different_pitches_and_instruments_are_independent() -> None:
    notes = [
        NoteEvent(0.0, 1.0, 60, 80),
        NoteEvent(0.0, 1.0, 61, 80),
        NoteEvent(0.0, 1.0, 60, 80, instrument="violin"),
    ]
    assert len(merge_overlapping(notes)) == 3


def test_merge_does_not_mutate_input() -> None:
    notes = [NoteEvent(0.0, 1.0, 60, 80), NoteEvent(0.01, 2.0, 60, 80)]
    merge_overlapping(notes)
    assert notes[0].offset == 1.0


def test_clean_notes_chain_and_stats() -> None:
    notes = [
        NoteEvent(0.0, 1.0, 60, 80),
        NoteEvent(0.01, 1.2, 60, 80),  # doublon
        NoteEvent(2.0, 2.01, 64, 80),  # trop courte
        NoteEvent(3.0, 4.0, 67, 3),  # trop faible
    ]
    cleaned, stats = clean_notes(notes, min_duration_s=0.03, min_velocity=12)
    assert [n.pitch for n in cleaned] == [60]
    assert stats == {"input": 4, "removed_low_velocity": 1, "merged": 1, "removed_short": 1, "output": 1}


def test_clean_notes_empty() -> None:
    cleaned, stats = clean_notes([])
    assert cleaned == [] and stats["output"] == 0
