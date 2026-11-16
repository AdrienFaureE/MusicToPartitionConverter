"""Notation : tonalité, métrique et construction MusicXML."""

from partition.notation.key_meter import (
    beat_length,
    detect_key,
    key_label,
    measure_length,
    parse_key,
    parse_time_signature,
    resolve_key,
)
from partition.notation.musicxml_builder import (
    VoiceEvent,
    assign_voices,
    build_score,
    spell_pitch,
    write_musicxml,
)

__all__ = [
    "VoiceEvent",
    "assign_voices",
    "beat_length",
    "build_score",
    "detect_key",
    "key_label",
    "measure_length",
    "parse_key",
    "parse_time_signature",
    "resolve_key",
    "spell_pitch",
    "write_musicxml",
]
