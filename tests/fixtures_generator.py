"""Génère les fixtures de test (MIDI et audio synthétiques) pour ne versionner aucun binaire.

Utilisation directe::

    python tests/fixtures_generator.py [dossier]

Les tests appellent :func:`generate_all` automatiquement si les fichiers sont absents.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

SAMPLE_RATE = 22050
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# (début en noires, durée en noires, hauteur MIDI)
Pattern = list[tuple[float, float, int]]


def tempo_pattern() -> Pattern:
    """Seize mesures : basse en noires, mélodie en noires avec quelques croches."""
    melody = [72, 74, 76, 77, 79, 77, 76, 74]
    pattern: Pattern = []
    for beat in range(64):
        pattern.append((float(beat), 1.0, 48 if beat % 4 == 0 else 55))
        pattern.append((float(beat), 1.0, melody[beat % len(melody)]))
        if beat % 8 == 7:
            pattern.append((beat + 0.5, 0.5, 72))
    return pattern


def piano_pattern() -> Pattern:
    """Dix secondes à 120 à la noire : accords à la main gauche, gamme à la main droite."""
    scale = [60, 62, 64, 65, 67, 69, 71, 72]
    pattern: Pattern = []
    for beat in range(20):
        pattern.append((float(beat), 1.0, scale[beat % 8] + 12))
        if beat % 2 == 0:
            for pitch in (48, 52, 55):
                pattern.append((float(beat), 2.0, pitch))
    return pattern


def pattern_to_midi(pattern: Pattern, bpm: float) -> pretty_midi.PrettyMIDI:
    """Convertit un motif en objet MIDI au tempo donné."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    piano = pretty_midi.Instrument(program=0, name="piano")
    beat = 60.0 / bpm
    for start, duration, pitch in pattern:
        piano.notes.append(pretty_midi.Note(80, pitch, start * beat, (start + 0.95 * duration) * beat))
    midi.instruments.append(piano)
    return midi


def synthesize(pattern: Pattern, bpm: float, sample_rate: int = SAMPLE_RATE, tail_s: float = 0.5) -> np.ndarray:
    """Synthèse additive simple (fondamentale et trois harmoniques, enveloppe décroissante)."""
    beat = 60.0 / bpm
    total = max(start + duration for start, duration, _ in pattern) * beat + tail_s
    audio = np.zeros(int(total * sample_rate), dtype=np.float64)
    for start, duration, pitch in pattern:
        length = int(duration * beat * sample_rate)
        t = np.arange(length) / sample_rate
        frequency = 440.0 * 2 ** ((pitch - 69) / 12)
        tone = sum(np.sin(2 * np.pi * frequency * h * t) / h**1.5 for h in (1, 2, 3, 4))
        envelope = np.exp(-3.0 * t) * np.minimum(1.0, t / 0.005)
        fade = np.minimum(1.0, (length - np.arange(length)) / (0.01 * sample_rate))
        begin = int(start * beat * sample_rate)
        audio[begin : begin + length] += tone * envelope * fade
    return (0.8 * audio / np.max(np.abs(audio))).astype(np.float32)


def generate_all(directory: Path = FIXTURES_DIR, force: bool = False) -> dict[str, Path]:
    """Crée toutes les fixtures et retourne leurs chemins."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "tempo_90": directory / "tempo_90.mid",
        "piano_mid": directory / "piano_10s.mid",
        "piano_wav": directory / "piano_10s.wav",
        "piano_flac": directory / "piano_10s.flac",
        "stereo_wav": directory / "stereo_44k.wav",
    }
    if not force and all(path.is_file() for path in paths.values()):
        return paths
    pattern_to_midi(tempo_pattern(), 90.0).write(str(paths["tempo_90"]))
    pattern_to_midi(piano_pattern(), 120.0).write(str(paths["piano_mid"]))
    audio = synthesize(piano_pattern(), 120.0)
    sf.write(str(paths["piano_wav"]), audio, SAMPLE_RATE, subtype="PCM_16")
    sf.write(str(paths["piano_flac"]), audio, SAMPLE_RATE)
    t = np.arange(44100) / 44100
    stereo = np.stack([0.5 * np.sin(2 * np.pi * 440 * t), 0.25 * np.sin(2 * np.pi * 660 * t)], axis=1)
    sf.write(str(paths["stereo_wav"]), stereo.astype(np.float32), 44100, subtype="PCM_16")
    return paths


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else FIXTURES_DIR
    for name, path in generate_all(target, force=True).items():
        print(f"{name}: {path}")
