"""Tonalité (détection Krumhansl-Schmuckler via music21) et métrique."""

from __future__ import annotations

import re
from fractions import Fraction

from music21 import key as m21key
from music21 import note as m21note
from music21 import stream as m21stream
from music21.analysis import discrete

from partition.postprocess.quantize import QuantizedNote

_KEY_PATTERN = re.compile(r"^\s*([A-Ga-g])\s*([#b♯♭-]?)\s*(major|minor|majeur|mineur|maj|min|m)?\s*$")
_TIME_PATTERN = re.compile(r"^\s*(\d{1,2})\s*/\s*(1|2|4|8|16|32)\s*$")


def parse_key(text: str) -> m21key.Key:
    """Interprète une tonalité écrite « C major », « a minor », « Eb major », « F# mineur »...

    Sans mode explicite, une tonique en minuscule désigne le mode mineur.

    Raises:
        ValueError: texte non reconnu.
    """
    match = _KEY_PATTERN.match(text)
    if not match:
        raise ValueError(f"Tonalité non reconnue : « {text} » (exemples : « C major », « a minor », « Eb major »).")
    letter, accidental, mode = match.groups()
    if mode is None:
        mode_name = "minor" if letter.islower() else "major"
    else:
        mode_name = "minor" if mode.lower() in ("minor", "mineur", "min", "m") else "major"
    alteration = {"#": "#", "♯": "#", "b": "-", "♭": "-", "-": "-", "": ""}[accidental]
    return m21key.Key(letter.upper() + alteration, mode_name)


def detect_key(notes: list[QuantizedNote]) -> m21key.Key:
    """Détecte la tonalité par l'algorithme de Krumhansl-Schmuckler.

    L'algorithme ne dépend que de la durée cumulée de chaque classe de hauteur : on lui fournit
    donc douze notes pondérées plutôt que la pièce entière, ce qui est équivalent et rapide.
    Retourne do majeur s'il n'y a aucune note.
    """
    totals = [0.0] * 12
    for note in notes:
        totals[note.pitch % 12] += float(note.duration)
    if not any(totals):
        return m21key.Key("C", "major")
    scale = 8.0 / max(totals)
    sample: m21stream.Stream = m21stream.Stream()
    for pitch_class, total in enumerate(totals):
        if total > 0:
            element = m21note.Note(60 + pitch_class)
            element.quarterLength = max(total * scale, 1e-3)
            sample.append(element)
    solution = discrete.KrumhanslSchmuckler().getSolution(sample)
    return m21key.Key(solution.tonic.name, solution.mode)


def resolve_key(notes: list[QuantizedNote], forced: str | None = None) -> tuple[m21key.Key, str]:
    """Retourne la tonalité et sa provenance (``forced`` ou ``detected``)."""
    if forced:
        return parse_key(forced), "forced"
    return detect_key(notes), "detected"


def key_label(key: m21key.Key) -> str:
    """Nom lisible d'une tonalité, par exemple « Eb major »."""
    return f"{key.tonic.name.replace('-', 'b')} {key.mode}"


def parse_time_signature(text: str) -> tuple[int, int]:
    """Interprète une métrique « 4/4 », « 3/4 », « 6/8 »...

    Raises:
        ValueError: texte non reconnu.
    """
    match = _TIME_PATTERN.match(text)
    if not match or int(match.group(1)) == 0:
        raise ValueError(f"Métrique non reconnue : « {text} » (exemples : « 4/4 », « 3/4 », « 6/8 »).")
    return int(match.group(1)), int(match.group(2))


def measure_length(time_signature: str) -> Fraction:
    """Longueur d'une mesure en noires (4/4 → 4, 6/8 → 3)."""
    numerator, denominator = parse_time_signature(time_signature)
    return Fraction(numerator * 4, denominator)


def beat_length(time_signature: str) -> Fraction:
    """Longueur d'un temps en noires (noire pointée pour les mesures composées comme 6/8)."""
    numerator, denominator = parse_time_signature(time_signature)
    if denominator >= 8 and numerator % 3 == 0 and numerator > 3:
        return Fraction(12, denominator)
    return Fraction(4, denominator)
