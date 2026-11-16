"""Estimation du tempo et de la grille de battements.

Deux méthodes :

- ``audio`` : suivi de battements ``librosa`` ; la grille peut suivre les battements détectés
  (rubato) ou rester à tempo constant ;
- ``midi`` : autocorrélation des attaques des notes transcrites, tempo constant.

L'unité de la grille est la noire : ``time_to_beat`` convertit des secondes en noires.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from partition.transcription.base import NoteEvent

FRAME_RATE = 100.0  # images par seconde de l'enveloppe d'attaques


@dataclass
class BeatGrid:
    """Grille de battements : instants (en secondes) de chaque noire."""

    bpm: float
    beat_times: list[float] = field(default_factory=list)
    method: str = "constant"

    def __post_init__(self) -> None:
        if len(self.beat_times) < 2:
            period = 60.0 / self.bpm
            start = self.beat_times[0] if self.beat_times else 0.0
            self.beat_times = [start, start + period]

    def time_to_beat(self, time_s: float) -> float:
        """Convertit un instant en position (noires), avec extrapolation linéaire aux extrémités."""
        times = self.beat_times
        if time_s <= times[0]:
            return (time_s - times[0]) / (times[1] - times[0])
        if time_s >= times[-1]:
            return len(times) - 1 + (time_s - times[-1]) / (times[-1] - times[-2])
        return float(np.interp(time_s, times, np.arange(len(times))))

    def beat_to_time(self, beat: float) -> float:
        """Convertit une position (noires) en instant."""
        times = self.beat_times
        last = len(times) - 1
        if beat <= 0:
            return times[0] + beat * (times[1] - times[0])
        if beat >= last:
            return times[-1] + (beat - last) * (times[-1] - times[-2])
        return float(np.interp(beat, np.arange(len(times)), times))


def constant_grid(bpm: float, start_s: float, end_s: float, method: str = "constant") -> BeatGrid:
    """Grille à tempo constant couvrant ``[start_s, end_s]``."""
    period = 60.0 / bpm
    count = max(2, int(math.ceil((end_s - start_s) / period)) + 2)
    return BeatGrid(bpm=bpm, beat_times=[start_s + i * period for i in range(count)], method=method)


def fold_bpm(bpm: float, min_bpm: float, max_bpm: float) -> float:
    """Ramène un tempo dans ``[min_bpm, max_bpm]`` par doublements ou divisions par deux."""
    if bpm <= 0:
        raise ValueError("Le tempo doit être strictement positif.")
    for _ in range(8):
        if bpm < min_bpm and bpm * 2 <= max_bpm * 1.0001:
            bpm *= 2
        elif bpm > max_bpm and bpm / 2 >= min_bpm * 0.9999:
            bpm /= 2
        else:
            break
    return bpm


def _onset_envelope(notes: list[NoteEvent]) -> np.ndarray:
    end = max(n.onset for n in notes)
    envelope = np.zeros(int(end * FRAME_RATE) + 8)
    for note in notes:
        envelope[int(round(note.onset * FRAME_RATE))] += note.velocity / 127.0
    kernel = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2)
    return np.convolve(envelope, kernel / kernel.sum(), mode="same")


def estimate_tempo_from_notes(
    notes: list[NoteEvent], min_bpm: float = 50.0, max_bpm: float = 180.0, default_bpm: float = 120.0
) -> float:
    """Estime un tempo constant par autocorrélation des attaques.

    Une préférence douce pour les tempos proches de 110 à la noire départage les multiples.
    Retourne ``default_bpm`` s'il y a trop peu de notes.
    """
    if len(notes) < 4:
        return default_bpm
    envelope = _onset_envelope(notes)
    envelope = envelope - envelope.mean()
    size = 1 << (2 * len(envelope) - 1).bit_length()
    spectrum = np.fft.rfft(envelope, size)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum))[: len(envelope)]
    if autocorr[0] <= 0:
        return default_bpm
    autocorr = autocorr / autocorr[0]
    lags = np.arange(len(autocorr))

    def value(lag: float) -> float:
        return float(np.interp(lag, lags, autocorr)) if lag < len(autocorr) - 1 else 0.0

    best_bpm, best_score = default_bpm, -math.inf
    for bpm in np.arange(min_bpm, max_bpm + 0.25, 0.25):
        lag = 60.0 / bpm * FRAME_RATE
        score = value(lag) + 0.5 * value(2 * lag) + 0.25 * value(4 * lag)
        score *= math.exp(-0.5 * (math.log2(bpm / 110.0) / 1.2) ** 2)
        if score > best_score:
            best_bpm, best_score = float(bpm), score
    return best_bpm


def _grid_phase(notes: list[NoteEvent], bpm: float, subdivision: int = 4) -> float:
    """Instant de départ de la grille : aligne les attaques sur les subdivisions du battement."""
    step = 60.0 / bpm / subdivision
    onsets = np.array([n.onset for n in notes])
    weights = np.array([n.velocity for n in notes], dtype=float)
    angles = 2 * np.pi * (onsets % step) / step
    mean_angle = math.atan2(float(np.sum(weights * np.sin(angles))), float(np.sum(weights * np.cos(angles))))
    phase = (mean_angle % (2 * np.pi)) / (2 * np.pi) * step
    first = float(onsets.min())
    return phase + round((first - phase) / step) * step


def estimate_grid_from_notes(
    notes: list[NoteEvent], min_bpm: float = 50.0, max_bpm: float = 180.0, forced_bpm: float | None = None
) -> BeatGrid:
    """Grille à tempo constant estimée depuis les notes (première note sur un battement)."""
    bpm = forced_bpm or estimate_tempo_from_notes(notes, min_bpm, max_bpm)
    if not notes:
        return constant_grid(bpm, 0.0, 1.0, method="midi")
    start = _grid_phase(notes, bpm)
    end = max(n.offset for n in notes)
    return constant_grid(bpm, start, end, method="forced" if forced_bpm else "midi")


def estimate_grid_from_audio(
    audio_path: Path,
    min_bpm: float = 50.0,
    max_bpm: float = 180.0,
    follow_beats: bool = True,
) -> BeatGrid:
    """Grille estimée par suivi de battements ``librosa``.

    Raises:
        ValueError: moins de quatre battements détectés (audio trop court ou sans pulsation).
    """
    import librosa

    samples, sample_rate = librosa.load(str(audio_path), sr=22050, mono=True)
    _, beats = librosa.beat.beat_track(y=samples, sr=sample_rate, units="time")
    beat_times = [float(t) for t in np.atleast_1d(beats)]
    if len(beat_times) < 4:
        raise ValueError("Trop peu de battements détectés dans l'audio.")
    bpm = 60.0 / float(np.median(np.diff(beat_times)))
    folded = fold_bpm(bpm, min_bpm, max_bpm)
    while folded > bpm * 1.5:  # tempo doublé : on insère les demi-battements
        doubled: list[float] = []
        for a, b in zip(beat_times[:-1], beat_times[1:], strict=True):
            doubled += [a, (a + b) / 2]
        beat_times = doubled + [beat_times[-1]]
        bpm *= 2
    while folded < bpm / 1.5:  # tempo divisé : un battement sur deux
        beat_times = beat_times[::2]
        bpm /= 2
    duration = len(samples) / sample_rate
    if not follow_beats:
        return constant_grid(bpm, beat_times[0], duration, method="audio-constant")
    return BeatGrid(bpm=bpm, beat_times=beat_times, method="audio-beats")


def estimate_grid(
    notes: list[NoteEvent],
    audio_path: Path | None = None,
    method: str = "audio",
    forced_bpm: float | None = None,
    min_bpm: float = 50.0,
    max_bpm: float = 180.0,
    follow_beats: bool = True,
) -> tuple[BeatGrid, list[str]]:
    """Choisit la méthode d'estimation et retourne la grille ainsi que d'éventuels avertissements."""
    warnings: list[str] = []
    if forced_bpm:
        return estimate_grid_from_notes(notes, min_bpm, max_bpm, forced_bpm=float(forced_bpm)), warnings
    if method == "audio" and audio_path is not None and Path(audio_path).is_file():
        try:
            return estimate_grid_from_audio(Path(audio_path), min_bpm, max_bpm, follow_beats), warnings
        except Exception as exc:
            warnings.append(f"Suivi de battements audio impossible ({exc}) : tempo estimé depuis les notes.")
    elif method == "audio":
        warnings.append("Audio absent : tempo estimé depuis les notes.")
    return estimate_grid_from_notes(notes, min_bpm, max_bpm), warnings
