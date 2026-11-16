"""Tests de la séparation des mains."""

from __future__ import annotations

import json
from fractions import Fraction as F
from pathlib import Path

import pytest

from partition.postprocess.hand_split import (
    HandOverrideError,
    load_overrides,
    parse_overrides,
    split_hands,
)
from partition.postprocess.quantize import QuantizedNote


def q(start: float, pitch: int, duration: float = 1.0, instrument: str = "piano") -> QuantizedNote:
    return QuantizedNote(F(start).limit_denominator(48), F(duration).limit_denominator(48), pitch, 80, instrument)


def pitches(notes: list[QuantizedNote]) -> list[int]:
    return sorted(n.pitch for n in notes)


def test_threshold_boundary() -> None:
    split = split_hands([q(0, 59), q(0, 60), q(0, 61)], split_pitch=60)
    assert pitches(split.left) == [59]
    assert pitches(split.right) == [60, 61]  # le do central va à la main droite


def test_threshold_splits_a_chord_across_the_threshold() -> None:
    split = split_hands([q(0, 57), q(0, 60), q(0, 64)], method="threshold")
    assert pitches(split.left) == [57] and pitches(split.right) == [60, 64]


def test_chord_method_keeps_a_playable_chord_together() -> None:
    # Accord à cheval sur le seuil, étendue d'une quinte : il reste entier, du côté de sa moyenne.
    high = split_hands([q(0, 57), q(0, 60), q(0, 64)], method="chord")
    assert pitches(high.right) == [57, 60, 64] and high.left == []
    low = split_hands([q(0, 52), q(0, 55), q(0, 60)], method="chord")
    assert pitches(low.left) == [52, 55, 60] and low.right == []


def test_chord_method_cuts_a_wide_chord_at_its_largest_gap() -> None:
    split = split_hands([q(0, 36), q(0, 43), q(0, 64), q(0, 67), q(0, 72)], method="chord")
    assert pitches(split.left) == [36, 43]
    assert pitches(split.right) == [64, 67, 72]


def test_chord_method_wide_chord_entirely_above_threshold() -> None:
    # Voix la plus haute à droite, la plus basse à gauche, même si tout est au-dessus du seuil.
    split = split_hands([q(0, 62), q(0, 84)], method="chord")
    assert pitches(split.left) == [62] and pitches(split.right) == [84]


def test_chord_method_isolated_notes_use_threshold() -> None:
    split = split_hands([q(0, 50), q(1, 70), q(2, 60)], method="chord")
    assert pitches(split.left) == [50] and pitches(split.right) == [60, 70]


def test_other_instruments_go_to_a_separate_track() -> None:
    split = split_hands([q(0, 72), q(0, 40, instrument="violin"), q(1, 90, instrument="flute")])
    assert pitches(split.right) == [72] and split.left == []
    assert sorted(n.instrument for n in split.others) == ["flute", "violin"]


def test_empty_input() -> None:
    split = split_hands([])
    assert split.right == split.left == split.others == []


def test_unknown_method() -> None:
    with pytest.raises(ValueError):
        split_hands([q(0, 60)], method="magie")


def test_override_global_split_and_method() -> None:
    overrides = parse_overrides({"split_pitch": 65})
    split = split_hands([q(0, 60), q(0, 64), q(0, 65)], overrides=overrides)
    assert pitches(split.left) == [60, 64] and pitches(split.right) == [65]
    forced_method = parse_overrides({"method": "chord"})
    split = split_hands([q(0, 57), q(0, 60), q(0, 64)], overrides=forced_method)
    assert split.left == []


def test_override_ranges_in_beats_and_measures() -> None:
    notes = [q(0, 62), q(4, 62), q(8, 62), q(12, 62)]
    overrides = parse_overrides(
        {
            "ranges": [
                {"start_measure": 2, "end_measure": 2, "split_pitch": 64},  # noires 4 à 8
                {"start_beat": 12, "end_beat": 16, "hand": "left"},
            ]
        }
    )
    split = split_hands(notes, overrides=overrides, beats_per_measure=4.0)
    assert sorted(float(n.start) for n in split.left) == [4.0, 12.0]
    assert sorted(float(n.start) for n in split.right) == [0.0, 8.0]


def test_override_single_note_wins() -> None:
    overrides = parse_overrides({"notes": [{"start_beat": 1.5, "pitch": 72, "hand": "left"}]})
    split = split_hands([q(1.5, 72), q(2, 72)], overrides=overrides)
    assert [float(n.start) for n in split.left] == [1.5]
    assert [float(n.start) for n in split.right] == [2.0]


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"inconnu": 1},
        {"method": "magie"},
        {"ranges": [{"start_beat": 0, "hand": "middle"}]},
        {"ranges": [{"start_beat": 0}]},
        {"ranges": [{"debut": 0, "hand": "left"}]},
        {"notes": [{"pitch": 60, "hand": "left"}]},
        {"notes": [{"start_beat": 0, "pitch": 60}]},
    ],
)
def test_invalid_overrides(data: object) -> None:
    with pytest.raises(HandOverrideError):
        parse_overrides(data)


def test_load_overrides_file(tmp_path: Path) -> None:
    path = tmp_path / "hand_overrides.json"
    path.write_text(json.dumps({"split_pitch": 55}), encoding="utf-8")
    assert load_overrides(path).split_pitch == 55
    path.write_text("{pas du json", encoding="utf-8")
    with pytest.raises(HandOverrideError, match="JSON invalide"):
        load_overrides(path)
    with pytest.raises(HandOverrideError, match="illisible"):
        load_overrides(tmp_path / "absent.json")
