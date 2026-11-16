"""Quantification des débuts et des durées sur la grille rythmique.

Les positions sont exprimées en noires, sous forme de fractions exactes.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from fractions import Fraction

from partition.postprocess.tempo import BeatGrid
from partition.transcription.base import PIANO, NoteEvent


@dataclass
class QuantizedNote:
    """Note alignée sur la grille (positions en noires)."""

    start: Fraction
    duration: Fraction
    pitch: int
    velocity: int
    instrument: str = PIANO

    @property
    def end(self) -> Fraction:
        """Fin de la note, en noires."""
        return self.start + self.duration


def grid_step(grid: int) -> Fraction:
    """Pas de la grille en noires (16 → une double croche = 1/4 de noire)."""
    if grid <= 0:
        raise ValueError("La grille doit être strictement positive.")
    return Fraction(4, grid)


def snap(value: float, grid: int = 16, triplets: bool = False) -> Fraction:
    """Aligne une position (en noires) sur la grille, avec les triolets de croches si demandé."""
    step = grid_step(grid)
    straight = Fraction(round(value / float(step))) * step
    if not triplets:
        return straight
    triplet = Fraction(round(value * 3), 3)
    return triplet if abs(float(triplet) - value) < abs(float(straight) - value) - 1e-9 else straight


def quantize_notes(
    notes: list[NoteEvent],
    grid: BeatGrid,
    grid_division: int = 16,
    triplets: bool = False,
    fill_gap_beats: float = 0.25,
) -> list[QuantizedNote]:
    """Quantifie les notes.

    - les débuts et les fins sont alignés sur la grille, la durée minimale est d'un pas ;
    - la pièce est décalée d'un nombre entier de noires pour commencer dans le premier battement ;
    - les doublons (même instrument, hauteur et début) sont fusionnés, et une note est raccourcie
      si la même hauteur est rejouée avant sa fin ;
    - un silence plus court que ``fill_gap_beats`` avant l'attaque suivante est absorbé par la note
      (jeu legato) ; les silences plus longs sont conservés.
    """
    if not notes:
        return []
    step = grid_step(grid_division)
    raw = [(grid.time_to_beat(n.onset), grid.time_to_beat(n.offset), n) for n in notes]
    shift = math.floor(min(start for start, _, _ in raw) + 1e-6)

    best: dict[tuple[str, int, Fraction], QuantizedNote] = {}
    for start_f, end_f, note in raw:
        start = snap(start_f - shift, grid_division, triplets)
        end = snap(end_f - shift, grid_division, triplets)
        if start < 0:
            start = Fraction(0)
        if end <= start:
            end = start + step
        candidate = QuantizedNote(start, end - start, note.pitch, note.velocity, note.instrument)
        key = (note.instrument, note.pitch, start)
        existing = best.get(key)
        if existing is None:
            best[key] = candidate
        else:
            existing.duration = max(existing.duration, candidate.duration)
            existing.velocity = max(existing.velocity, candidate.velocity)

    result = sorted(best.values(), key=lambda n: (n.start, n.pitch, n.instrument))

    # Une même hauteur rejouée coupe la note précédente.
    last_by_pitch: dict[tuple[str, int], QuantizedNote] = {}
    for current in result:
        pitch_key = (current.instrument, current.pitch)
        previous = last_by_pitch.get(pitch_key)
        if previous is not None and previous.end > current.start:
            previous.duration = current.start - previous.start
        last_by_pitch[pitch_key] = current

    # Absorption des petits silences (par instrument).
    fill = Fraction(fill_gap_beats).limit_denominator(96)
    if fill > 0:
        onsets_by_instrument: dict[str, list[Fraction]] = {}
        for current in result:
            onsets_by_instrument.setdefault(current.instrument, []).append(current.start)
        for instrument, values in onsets_by_instrument.items():
            onsets_by_instrument[instrument] = sorted(set(values))
        for current in result:
            onsets = onsets_by_instrument[current.instrument]
            index = bisect.bisect_left(onsets, current.end)
            if index < len(onsets):
                gap = onsets[index] - current.end
                if 0 < gap <= fill:
                    current.duration += gap
    return [n for n in result if n.duration > 0]
