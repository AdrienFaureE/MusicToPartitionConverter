"""Séparation main droite / main gauche.

Deux méthodes automatiques :

- ``threshold`` : une note va à la main droite si sa hauteur est supérieure ou égale au seuil ;
- ``chord`` : les notes attaquées ensemble sont traitées comme un accord. Un accord jouable d'une
  seule main (étendue <= ``max_hand_span``) reste entier, du côté de sa hauteur moyenne ; un accord
  plus large est coupé à son plus grand intervalle (voix hautes à droite, voix basses à gauche).

Un fichier JSON d'override par job permet de forcer la séparation (voir :func:`load_overrides`).
Les notes des autres instruments sont rangées à part.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from partition.postprocess.quantize import QuantizedNote
from partition.transcription.base import PIANO

RIGHT = "right"
LEFT = "left"
OVERRIDES_FILENAME = "hand_overrides.json"


class HandOverrideError(ValueError):
    """Fichier d'override invalide."""


@dataclass
class RangeOverride:
    """Règle appliquée sur un intervalle de temps (en noires ou en mesures, bornes incluses pour les mesures)."""

    start_beat: float | None = None
    end_beat: float | None = None
    start_measure: int | None = None
    end_measure: int | None = None
    split_pitch: int | None = None
    hand: str | None = None

    def bounds(self, beats_per_measure: float) -> tuple[float, float]:
        """Intervalle ``[début, fin[`` en noires."""
        if self.start_measure is not None or self.end_measure is not None:
            first = (self.start_measure or 1) - 1
            last = self.end_measure if self.end_measure is not None else 10**9
            return first * beats_per_measure, last * beats_per_measure
        return (
            self.start_beat if self.start_beat is not None else 0.0,
            self.end_beat if self.end_beat is not None else float("inf"),
        )


@dataclass
class NoteOverride:
    """Main imposée pour une note précise (début en noires, hauteur MIDI)."""

    start_beat: float
    pitch: int
    hand: str


@dataclass
class HandOverrides:
    """Contenu d'un fichier d'override."""

    split_pitch: int | None = None
    method: str | None = None
    ranges: list[RangeOverride] = field(default_factory=list)
    notes: list[NoteOverride] = field(default_factory=list)


@dataclass
class HandSplit:
    """Résultat de la séparation."""

    right: list[QuantizedNote] = field(default_factory=list)
    left: list[QuantizedNote] = field(default_factory=list)
    others: list[QuantizedNote] = field(default_factory=list)


def _check_hand(value: Any, context: str) -> str:
    if value not in (RIGHT, LEFT):
        raise HandOverrideError(f"{context} : « hand » doit valoir « right » ou « left » (reçu : {value!r}).")
    return str(value)


def parse_overrides(data: Any) -> HandOverrides:
    """Valide et convertit le contenu JSON d'un fichier d'override.

    Format attendu (toutes les clés sont optionnelles)::

        {
          "split_pitch": 57,
          "method": "chord",
          "ranges": [
            {"start_measure": 5, "end_measure": 8, "split_pitch": 64},
            {"start_beat": 32, "end_beat": 36, "hand": "left"}
          ],
          "notes": [{"start_beat": 4.5, "pitch": 60, "hand": "left"}]
        }
    """
    if not isinstance(data, dict):
        raise HandOverrideError("Le fichier d'override doit contenir un objet JSON.")
    unknown = set(data) - {"split_pitch", "method", "ranges", "notes"}
    if unknown:
        raise HandOverrideError(f"Clé(s) inconnue(s) dans l'override : {', '.join(sorted(unknown))}.")
    method = data.get("method")
    if method not in (None, "threshold", "chord"):
        raise HandOverrideError("« method » doit valoir « threshold » ou « chord ».")
    overrides = HandOverrides(split_pitch=data.get("split_pitch"), method=method)
    try:
        for index, item in enumerate(data.get("ranges") or []):
            rule = RangeOverride(**item)
            if rule.hand is not None:
                _check_hand(rule.hand, f"ranges[{index}]")
            if rule.hand is None and rule.split_pitch is None:
                raise HandOverrideError(f"ranges[{index}] : indiquer « split_pitch » ou « hand ».")
            overrides.ranges.append(rule)
        for index, item in enumerate(data.get("notes") or []):
            note = NoteOverride(float(item["start_beat"]), int(item["pitch"]), item.get("hand"))
            _check_hand(note.hand, f"notes[{index}]")
            overrides.notes.append(note)
    except (TypeError, KeyError) as exc:
        raise HandOverrideError(f"Override mal formé : {exc}") from exc
    return overrides


def load_overrides(path: str | Path) -> HandOverrides:
    """Charge un fichier JSON d'override."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HandOverrideError(f"Fichier d'override illisible : {path} ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise HandOverrideError(f"JSON invalide dans {path} : {exc}") from exc
    return parse_overrides(data)


def _split_chord(chord: list[QuantizedNote], split_pitch: int, max_hand_span: int) -> dict[int, str]:
    """Attribue une main à chaque note d'un accord (clé : ``id`` de la note)."""
    ordered = sorted(chord, key=lambda n: n.pitch)
    if len(ordered) == 1:
        return {id(ordered[0]): RIGHT if ordered[0].pitch >= split_pitch else LEFT}
    span = ordered[-1].pitch - ordered[0].pitch
    if span <= max_hand_span:
        mean = sum(n.pitch for n in ordered) / len(ordered)
        hand = RIGHT if mean >= split_pitch else LEFT
        return {id(n): hand for n in ordered}
    gaps = [ordered[i + 1].pitch - ordered[i].pitch for i in range(len(ordered) - 1)]
    cut = max(range(len(gaps)), key=lambda i: (gaps[i], -abs(i - (len(gaps) - 1) / 2)))
    return {id(n): (LEFT if i <= cut else RIGHT) for i, n in enumerate(ordered)}


def split_hands(
    notes: list[QuantizedNote],
    method: str = "threshold",
    split_pitch: int = 60,
    max_hand_span: int = 14,
    overrides: HandOverrides | None = None,
    beats_per_measure: float = 4.0,
) -> HandSplit:
    """Répartit les notes entre main droite, main gauche et autres instruments."""
    overrides = overrides or HandOverrides()
    method = overrides.method or method
    if method not in ("threshold", "chord"):
        raise ValueError(f"Méthode de séparation inconnue : {method}")
    base_split = overrides.split_pitch if overrides.split_pitch is not None else split_pitch
    bounds = [(rule, rule.bounds(beats_per_measure)) for rule in overrides.ranges]

    def rule_for(start: Fraction) -> RangeOverride | None:
        position = float(start)
        for rule, (low, high) in reversed(bounds):  # la dernière règle déclarée l'emporte
            if low <= position < high:
                return rule
        return None

    result = HandSplit()
    piano = [n for n in notes if n.instrument == PIANO]
    result.others = [n for n in notes if n.instrument != PIANO]

    hands: dict[int, str] = {}
    chords: dict[Fraction, list[QuantizedNote]] = defaultdict(list)
    for note in piano:
        chords[note.start].append(note)
    for start, chord in chords.items():
        rule = rule_for(start)
        if rule is not None and rule.hand is not None:
            hands.update({id(n): rule.hand for n in chord})
            continue
        local_split = rule.split_pitch if rule is not None and rule.split_pitch is not None else base_split
        if method == "chord":
            hands.update(_split_chord(chord, local_split, max_hand_span))
        else:
            hands.update({id(n): RIGHT if n.pitch >= local_split else LEFT for n in chord})

    for forced in overrides.notes:
        for note in piano:
            if note.pitch == forced.pitch and abs(float(note.start) - forced.start_beat) < 1e-3:
                hands[id(note)] = forced.hand

    for note in sorted(piano, key=lambda n: (n.start, n.pitch)):
        (result.right if hands[id(note)] == RIGHT else result.left).append(note)
    return result
