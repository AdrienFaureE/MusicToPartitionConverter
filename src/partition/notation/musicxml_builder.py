"""Construction de la partition music21 (deux portées piano) et export MusicXML.

Les mesures sont construites explicitement : découpage des notes aux barres de mesure avec
liaisons, remplissage des silences, répartition en voix quand des notes se chevauchent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from music21 import chord as m21chord
from music21 import clef as m21clef
from music21 import instrument as m21instrument
from music21 import key as m21key
from music21 import layout as m21layout
from music21 import metadata as m21metadata
from music21 import meter as m21meter
from music21 import note as m21note
from music21 import pitch as m21pitch
from music21 import stream as m21stream
from music21 import tempo as m21tempo
from music21 import tie as m21tie

from partition.notation.key_meter import beat_length, measure_length
from partition.postprocess.hand_split import HandSplit
from partition.postprocess.quantize import QuantizedNote

_SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_NAMES = ["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"]


@dataclass
class VoiceEvent:
    """Note ou accord d'une voix (positions en noires)."""

    start: Fraction
    duration: Fraction
    pitches: list[int] = field(default_factory=list)
    velocity: int = 64

    @property
    def end(self) -> Fraction:
        """Fin de l'événement."""
        return self.start + self.duration


@dataclass
class _Segment:
    offset: Fraction  # position dans la mesure
    duration: Fraction
    event: VoiceEvent | None  # None = silence
    tie: str | None = None
    hidden: bool = False


def spell_pitch(midi: int, sharps: int = 0) -> m21pitch.Pitch:
    """Orthographie une hauteur MIDI : bémols dans les tonalités à bémols, dièses sinon."""
    names = _FLAT_NAMES if sharps < 0 else _SHARP_NAMES
    return m21pitch.Pitch(f"{names[midi % 12]}{midi // 12 - 1}")


def assign_voices(notes: list[QuantizedNote], max_voices: int = 2) -> list[list[VoiceEvent]]:
    """Regroupe les notes en accords et les répartit en voix sans chevauchement.

    Les notes de même début et même durée forment un accord. Un accord qui chevauche la voix 1
    passe en voix 2, etc. Quand toutes les voix sont occupées, la note précédente de la voix la
    plus proche de se libérer est raccourcie ; si deux accords commencent ensemble, ils sont fusionnés.
    """
    groups: dict[tuple[Fraction, Fraction], VoiceEvent] = {}
    for note in sorted(notes, key=lambda n: (n.start, -n.duration, n.pitch)):
        key = (note.start, note.duration)
        event = groups.setdefault(key, VoiceEvent(note.start, note.duration))
        if note.pitch not in event.pitches:
            event.pitches.append(note.pitch)
        event.velocity = max(event.velocity if len(event.pitches) > 1 else 0, note.velocity)
    events = sorted(groups.values(), key=lambda e: (e.start, -e.duration))

    voices: list[list[VoiceEvent]] = [[] for _ in range(max(1, max_voices))]
    for event in events:
        target = next((v for v in voices if not v or v[-1].end <= event.start), None)
        if target is not None:
            target.append(event)
            continue
        cuttable = [v for v in voices if v[-1].start < event.start]
        if cuttable:
            voice = min(cuttable, key=lambda v: v[-1].end)
            voice[-1].duration = event.start - voice[-1].start
            voice.append(event)
        else:  # même début dans toutes les voix : fusion avec la durée la plus proche
            voice = min(voices, key=lambda v: abs(v[-1].duration - event.duration))
            voice[-1].pitches = sorted(set(voice[-1].pitches) | set(event.pitches))
            voice[-1].velocity = max(voice[-1].velocity, event.velocity)
    for voice in voices:
        for event in voice:
            event.pitches.sort()
    return [v for i, v in enumerate(voices) if v or i == 0]


def _split_rest(offset: Fraction, duration: Fraction, beat: Fraction) -> list[tuple[Fraction, Fraction]]:
    """Découpe un silence aux temps de la mesure pour rester lisible."""
    end = offset + duration
    if offset % beat == 0 and duration % beat == 0:
        return [(offset, duration)]
    pieces: list[tuple[Fraction, Fraction]] = []
    position = offset
    while position < end:
        boundary = (position // beat + 1) * beat
        if position % beat == 0:  # on avale tous les temps entiers disponibles
            whole = ((end - position) // beat) * beat
            boundary = position + whole if whole > 0 else end
        stop = min(boundary, end)
        pieces.append((position, stop - position))
        position = stop
    return pieces


def _measure_segments(
    voice: list[VoiceEvent], measure_len: Fraction, measure_count: int, beat: Fraction, primary: bool
) -> list[list[_Segment]]:
    """Répartit une voix dans les mesures, avec liaisons et silences."""
    per_measure: list[list[_Segment]] = [[] for _ in range(measure_count)]
    for event in voice:
        position = event.start
        parts: list[tuple[int, Fraction, Fraction]] = []
        while position < event.end:
            index = int(position // measure_len)
            stop = min(event.end, (index + 1) * measure_len)
            parts.append((index, position - index * measure_len, stop - position))
            position = stop
        for number, (index, offset, duration) in enumerate(parts):
            tie = None
            if len(parts) > 1:
                tie = "start" if number == 0 else ("stop" if number == len(parts) - 1 else "continue")
            per_measure[index].append(_Segment(offset, duration, event, tie))

    for index, segments in enumerate(per_measure):
        if not segments and not primary:
            continue
        segments.sort(key=lambda s: s.offset)
        filled: list[_Segment] = []
        cursor = Fraction(0)
        for segment in segments + [_Segment(measure_len, Fraction(0), None)]:
            if segment.offset > cursor:
                gap = segment.offset - cursor
                if cursor == 0 and gap == measure_len:
                    filled.append(_Segment(cursor, gap, None, hidden=not primary))
                else:
                    for offset, duration in _split_rest(cursor, gap, beat):
                        filled.append(_Segment(offset, duration, None, hidden=not primary))
            if segment.duration > 0:
                filled.append(segment)
                cursor = segment.offset + segment.duration
        per_measure[index] = filled
    return per_measure


def _to_element(segment: _Segment, sharps: int) -> m21note.GeneralNote:
    element: m21note.GeneralNote
    if segment.event is None:
        element = m21note.Rest()
        if segment.hidden:
            element.style.hideObjectOnPrint = True
    else:
        pitches = [spell_pitch(p, sharps) for p in segment.event.pitches]
        element = m21note.Note(pitches[0]) if len(pitches) == 1 else m21chord.Chord(pitches)
        element.volume.velocity = segment.event.velocity
        if segment.tie:
            element.tie = m21tie.Tie(segment.tie)
    element.quarterLength = segment.duration
    return element


def _build_staff(
    notes: list[QuantizedNote],
    staff_id: str,
    clef: m21clef.Clef,
    key: m21key.Key,
    time_signature: str,
    measure_count: int,
    max_voices: int,
    bpm: float | None,
) -> m21stream.PartStaff:
    measure_len = measure_length(time_signature)
    beat = beat_length(time_signature)
    staff = m21stream.PartStaff(id=staff_id)
    staff.partName = "Piano"
    staff.insert(0, m21instrument.Piano())
    voices = assign_voices(notes, max_voices)
    segments = [
        _measure_segments(voice, measure_len, measure_count, beat, primary=(index == 0))
        for index, voice in enumerate(voices)
    ]
    for index in range(measure_count):
        measure = m21stream.Measure(number=index + 1)
        if index == 0:
            measure.insert(0, clef)
            measure.insert(0, m21key.KeySignature(key.sharps))
            measure.insert(0, m21meter.TimeSignature(time_signature))
            if bpm:
                measure.insert(0, m21tempo.MetronomeMark(number=round(bpm)))
        active = [voice_segments[index] for voice_segments in segments if voice_segments[index]]
        if len(active) == 1:
            for segment in active[0]:
                measure.insert(segment.offset, _to_element(segment, key.sharps))
        else:
            for number, voice_segments in enumerate(active, start=1):
                voice = m21stream.Voice(id=str(number))
                for segment in voice_segments:
                    voice.insert(segment.offset, _to_element(segment, key.sharps))
                measure.insert(0, voice)
        staff.append(measure)
    return staff


def build_score(
    split: HandSplit,
    key: m21key.Key | None = None,
    time_signature: str = "4/4",
    bpm: float | None = None,
    title: str | None = None,
    composer: str | None = None,
    max_voices_per_staff: int = 2,
    author: str | None = None,
) -> m21stream.Score:
    """Construit la partition piano : portée en clef de sol (main droite) et en clef de fa (main gauche)."""
    key = key or m21key.Key("C", "major")
    measure_len = measure_length(time_signature)
    last = max((n.end for n in split.right + split.left), default=Fraction(0))
    measure_count = max(1, math.ceil(last / measure_len))

    right = _build_staff(
        split.right, "RH", m21clef.TrebleClef(), key, time_signature, measure_count, max_voices_per_staff, bpm
    )
    left = _build_staff(
        split.left, "LH", m21clef.BassClef(), key, time_signature, measure_count, max_voices_per_staff, None
    )
    score = m21stream.Score(id="partition")
    score.metadata = m21metadata.Metadata()
    score.metadata.title = title or "Partition"
    if composer:
        score.metadata.composer = composer
    if author:
        score.metadata.arranger = author
    score.insert(0, right)
    score.insert(0, left)
    score.insert(0, m21layout.StaffGroup([right, left], name="Piano", abbreviation="Pno.", symbol="brace"))
    return score


def write_musicxml(score: m21stream.Score, path: str | Path) -> Path:
    """Écrit la partition au format MusicXML (non compressé)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = Path(score.write("musicxml", fp=str(path)))
    if written != path:  # music21 peut imposer son extension
        written.replace(path)
    return path
