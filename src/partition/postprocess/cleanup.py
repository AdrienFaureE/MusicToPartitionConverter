"""Nettoyage des notes brutes : notes trop courtes, doublons, chevauchements, vélocités faibles."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from partition.transcription.base import NoteEvent


def remove_short_notes(notes: list[NoteEvent], min_duration_s: float = 0.03) -> list[NoteEvent]:
    """Supprime les notes strictement plus courtes que ``min_duration_s``."""
    return [n for n in notes if n.duration >= min_duration_s]


def apply_velocity_threshold(notes: list[NoteEvent], min_velocity: int = 12) -> list[NoteEvent]:
    """Supprime les notes dont la vélocité est inférieure au seuil (0 désactive le filtre)."""
    if min_velocity <= 0:
        return list(notes)
    return [n for n in notes if n.velocity >= min_velocity]


def merge_overlapping(notes: list[NoteEvent], merge_window_s: float = 0.05) -> list[NoteEvent]:
    """Résout les chevauchements entre notes de même hauteur (et même instrument).

    - Si la seconde note commence moins de ``merge_window_s`` après la première, c'est un doublon :
      les deux sont fusionnées (début de la première, fin la plus tardive, vélocité la plus forte).
    - Si elle commence plus tard alors que la première sonne encore, c'est une note rejouée : la
      première est raccourcie à l'attaque de la seconde, et les deux sont conservées.

    Deux notes qui se touchent sans se chevaucher (note répétée) ne sont jamais fusionnées.
    """
    groups: dict[tuple[str, int], list[NoteEvent]] = defaultdict(list)
    for note in notes:
        groups[(note.instrument, note.pitch)].append(note)
    merged: list[NoteEvent] = []
    for group in groups.values():
        group.sort(key=lambda n: (n.onset, n.offset))
        current = replace(group[0])
        for note in group[1:]:
            if note.onset < current.offset - 1e-9 and note.onset - current.onset <= merge_window_s:
                current.offset = max(current.offset, note.offset)
                current.velocity = max(current.velocity, note.velocity)
                if note.confidence is not None:
                    current.confidence = max(current.confidence or 0.0, note.confidence)
                continue
            if note.onset < current.offset:
                current.offset = note.onset
            merged.append(current)
            current = replace(note)
        merged.append(current)
    merged.sort(key=lambda n: (n.onset, n.pitch))
    return merged


def clean_notes(
    notes: list[NoteEvent],
    min_duration_s: float = 0.03,
    min_velocity: int = 12,
    merge_window_s: float = 0.05,
) -> tuple[list[NoteEvent], dict[str, int]]:
    """Applique toute la chaîne de nettoyage.

    Returns:
        Les notes nettoyées et des statistiques (nombre de notes retirées par étape).
    """
    after_velocity = apply_velocity_threshold(notes, min_velocity)
    after_merge = merge_overlapping(after_velocity, merge_window_s)
    after_short = remove_short_notes(after_merge, min_duration_s)
    stats = {
        "input": len(notes),
        "removed_low_velocity": len(notes) - len(after_velocity),
        "merged": len(after_velocity) - len(after_merge),
        "removed_short": len(after_merge) - len(after_short),
        "output": len(after_short),
    }
    return after_short, stats
