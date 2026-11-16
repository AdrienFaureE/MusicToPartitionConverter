"""Lecture et écriture MIDI (``pretty_midi``) pour les notes brutes et quantifiées.

Ce module n'était pas dans l'arborescence demandée : il regroupe les conversions MIDI utilisées à
la fois par le pipeline, la reprise à partir d'une étape et les tests.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pretty_midi

from partition.postprocess.quantize import QuantizedNote, snap
from partition.transcription.base import PIANO, NoteEvent, normalize_events
from partition.transcription.mt3_model import program_to_instrument

_PROGRAMS = {name.lower(): index for index, name in enumerate(pretty_midi.INSTRUMENT_MAP)}


def instrument_to_program(instrument: str) -> tuple[int, bool]:
    """Retourne ``(programme General MIDI, est_percussion)`` pour un nom d'instrument."""
    if instrument == "drums":
        return 0, True
    if instrument.startswith("program_"):
        try:
            return int(instrument.split("_", 1)[1]), False
        except ValueError:
            return 0, False
    return _PROGRAMS.get(instrument.lower(), 0), False


def _new_instrument(name: str) -> pretty_midi.Instrument:
    program, is_drum = instrument_to_program(name)
    return pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)


def notes_to_midi(notes: list[NoteEvent], path: str | Path, bpm: float = 120.0) -> Path:
    """Écrit des notes (temps en secondes) dans un fichier MIDI, une piste par instrument."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    tracks: dict[str, pretty_midi.Instrument] = {}
    for note in notes:
        track = tracks.setdefault(note.instrument, _new_instrument(note.instrument))
        track.notes.append(pretty_midi.Note(note.velocity, note.pitch, note.onset, note.offset))
    midi.instruments.extend(tracks.values())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    return path


def _instrument_name(track: pretty_midi.Instrument) -> str:
    name = (track.name or "").strip().lower()
    if name == PIANO or name in _PROGRAMS or name == "drums" or name.startswith("program_"):
        return name
    return program_to_instrument(track.program, track.is_drum)


def midi_to_notes(path: str | Path) -> list[NoteEvent]:
    """Lit un fichier MIDI et retourne ses notes normalisées."""
    midi = pretty_midi.PrettyMIDI(str(path))
    events = [
        NoteEvent(note.start, note.end, note.pitch, note.velocity, _instrument_name(track))
        for track in midi.instruments
        for note in track.notes
    ]
    return normalize_events(events)


def midi_tempo(path: str | Path, default: float = 120.0) -> float:
    """Premier tempo déclaré dans un fichier MIDI."""
    _, tempi = pretty_midi.PrettyMIDI(str(path)).get_tempo_changes()
    return float(tempi[0]) if len(tempi) else default


def quantized_to_midi(notes: list[QuantizedNote], path: str | Path, bpm: float) -> Path:
    """Écrit des notes quantifiées à tempo constant (résolution 480 : doubles croches et triolets exacts)."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(bpm), resolution=480)
    seconds_per_beat = 60.0 / float(bpm)
    tracks: dict[str, pretty_midi.Instrument] = {}
    for note in notes:
        if note.instrument not in tracks:
            tracks[note.instrument] = _new_instrument(note.instrument)
        tracks[note.instrument].notes.append(
            pretty_midi.Note(
                note.velocity,
                note.pitch,
                float(note.start) * seconds_per_beat,
                float(note.end) * seconds_per_beat,
            )
        )
    midi.instruments.extend(tracks.values())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    return path


def midi_to_quantized(
    path: str | Path, grid: int = 16, triplets: bool = False
) -> tuple[list[QuantizedNote], float]:
    """Relit un ``quantized.mid`` : retourne les notes sur la grille et le tempo."""
    bpm = midi_tempo(path)
    beats_per_second = bpm / 60.0
    result: list[QuantizedNote] = []
    for event in midi_to_notes(path):
        start = snap(event.onset * beats_per_second, grid, triplets)
        end = snap(event.offset * beats_per_second, grid, triplets)
        if end <= start:
            end = start + Fraction(4, grid)
        result.append(QuantizedNote(start, end - start, event.pitch, event.velocity, event.instrument))
    result.sort(key=lambda n: (n.start, n.pitch))
    return result, bpm
