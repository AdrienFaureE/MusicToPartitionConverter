"""Tests de la notation : tonalité, métrique, construction et validité du MusicXML."""

from __future__ import annotations

from fractions import Fraction as F
from pathlib import Path

import pytest
from music21 import converter

from partition.midi_io import (
    instrument_to_program,
    midi_tempo,
    midi_to_notes,
    midi_to_quantized,
    notes_to_midi,
    quantized_to_midi,
)
from partition.notation import (
    assign_voices,
    beat_length,
    build_score,
    detect_key,
    key_label,
    measure_length,
    parse_key,
    parse_time_signature,
    resolve_key,
    spell_pitch,
    write_musicxml,
)
from partition.postprocess.hand_split import HandSplit, split_hands
from partition.postprocess.quantize import QuantizedNote
from partition.render import export_pdf, find_lilypond, find_musescore, render_html, render_svg_pages
from partition.render.html_render import RenderError
from partition.transcription import NoteEvent


def q(start: float, duration: float, pitch: int, instrument: str = "piano") -> QuantizedNote:
    return QuantizedNote(F(start).limit_denominator(48), F(duration).limit_denominator(48), pitch, 80, instrument)


def c_major_piece() -> list[QuantizedNote]:
    scale = [60, 62, 64, 65, 67, 69, 71, 72]
    notes = [q(i, 1, pitch + 12) for i, pitch in enumerate(scale)]
    notes += [q(0, 4, 48), q(0, 4, 52), q(0, 4, 55), q(4, 4, 41), q(4, 4, 45), q(4, 4, 48)]
    return notes


# --- tonalité et métrique --------------------------------------------------
def test_detect_key_c_major_and_a_minor() -> None:
    assert key_label(detect_key(c_major_piece())) == "C major"
    a_minor = [q(i, 1, p) for i, p in enumerate([57, 59, 60, 62, 64, 65, 68, 69, 57, 64, 57, 60, 64, 57])]
    assert key_label(detect_key(a_minor)) == "A minor"
    assert key_label(detect_key([])) == "C major"


@pytest.mark.parametrize(
    ("text", "label", "sharps"),
    [("C major", "C major", 0), ("a minor", "A minor", 0), ("Eb major", "Eb major", -3),
     ("F# mineur", "F# minor", 3), ("d", "D minor", -1), ("G", "G major", 1)],
)  # fmt: skip
def test_parse_key(text: str, label: str, sharps: int) -> None:
    key = parse_key(text)
    assert key_label(key) == label and key.sharps == sharps


def test_parse_key_invalid() -> None:
    with pytest.raises(ValueError):
        parse_key("H dur")


def test_resolve_key_forced_or_detected() -> None:
    key, source = resolve_key(c_major_piece(), "Eb major")
    assert (key_label(key), source) == ("Eb major", "forced")
    key, source = resolve_key(c_major_piece(), None)
    assert (key_label(key), source) == ("C major", "detected")


def test_time_signature_helpers() -> None:
    assert parse_time_signature("3/4") == (3, 4)
    assert measure_length("4/4") == 4 and measure_length("6/8") == 3 and measure_length("3/4") == 3
    assert beat_length("4/4") == 1 and beat_length("6/8") == F(3, 2) and beat_length("3/8") == F(1, 2)
    for invalid in ("4-4", "0/4", "4/5", "quatre"):
        with pytest.raises(ValueError):
            parse_time_signature(invalid)


def test_spell_pitch_follows_key() -> None:
    assert spell_pitch(61, sharps=2).nameWithOctave == "C#4"
    assert spell_pitch(61, sharps=-3).nameWithOctave == "D-4"
    assert spell_pitch(60).midi == 60


# --- voix --------------------------------------------------------------------
def test_assign_voices_groups_chords() -> None:
    voices = assign_voices([q(0, 1, 60), q(0, 1, 64), q(1, 1, 62)], max_voices=2)
    assert len(voices) == 1
    assert [e.pitches for e in voices[0]] == [[60, 64], [62]]


def test_assign_voices_uses_second_voice_for_overlap() -> None:
    voices = assign_voices([q(0, 4, 48), q(1, 1, 55), q(2, 1, 57)], max_voices=2)
    assert len(voices) == 2
    assert [e.pitches for e in voices[0]] == [[48]]
    assert [e.pitches for e in voices[1]] == [[55], [57]]


def test_assign_voices_truncates_when_all_voices_are_busy() -> None:
    voices = assign_voices([q(0, 4, 48), q(1, 4, 55), q(2, 1, 60)], max_voices=2)
    for voice in voices:
        for first, second in zip(voice, voice[1:], strict=False):
            assert first.end <= second.start
    assert sorted(p for voice in voices for e in voice for p in e.pitches) == [48, 55, 60]


def test_assign_voices_single_voice_merges_simultaneous_groups() -> None:
    voices = assign_voices([q(0, 4, 48), q(0, 1, 60)], max_voices=1)
    assert len(voices) == 1 and voices[0][0].pitches == [48, 60]


# --- partition ---------------------------------------------------------------
def test_score_structure_and_musicxml_validity(tmp_path: Path) -> None:
    split = split_hands(c_major_piece())
    score = build_score(split, time_signature="4/4", bpm=96, title="Gamme", composer="Anonyme")
    path = write_musicxml(score, tmp_path / "score.musicxml")
    text = path.read_text(encoding="utf-8")
    assert text.count("<staves>2</staves>") == 1
    assert "<sign>G</sign>" in text and "<sign>F</sign>" in text
    assert "Gamme" in text

    parsed = converter.parse(str(path))
    assert len(parsed.parts) == 2
    for part in parsed.parts:
        measures = part.getElementsByClass("Measure")
        assert len(measures) == 2
        for measure in measures:
            assert measure.duration.quarterLength == 4
    right, left = parsed.parts
    assert [n.pitch.midi for n in right.flatten().notes] == [72, 74, 76, 77, 79, 81, 83, 84]
    assert [sorted(p.midi for p in c.pitches) for c in left.flatten().notes] == [[48, 52, 55], [41, 45, 48]]
    assert right.flatten().getElementsByClass("TimeSignature")[0].ratioString == "4/4"
    assert right.flatten().getElementsByClass("MetronomeMark")[0].number == 96


def test_notes_across_barlines_are_tied_and_rests_fill_measures(tmp_path: Path) -> None:
    split = HandSplit(right=[q(3, 2, 72), q(9.5, 0.5, 74)], left=[])
    path = write_musicxml(build_score(split), tmp_path / "ties.musicxml")
    parsed = converter.parse(str(path))
    right, left = parsed.parts
    measures = list(right.getElementsByClass("Measure"))
    assert len(measures) == 3
    tied = [n for n in right.flatten().notes if n.pitch.midi == 72]
    assert [n.tie.type for n in tied] == ["start", "stop"]
    assert sum(float(n.quarterLength) for n in tied) == 2
    for part in (right, left):
        for measure in part.getElementsByClass("Measure"):
            total = sum(float(e.quarterLength) for e in measure.flatten().notesAndRests)
            assert total == pytest.approx(4.0)
    assert len(left.flatten().notes) == 0  # portée vide : silences uniquement


def test_key_signature_and_meter_are_written(tmp_path: Path) -> None:
    split = split_hands([q(0, 1, 63), q(1, 1, 70), q(2, 1, 51)])
    path = write_musicxml(build_score(split, key=parse_key("Eb major"), time_signature="3/4"), tmp_path / "k.musicxml")
    parsed = converter.parse(str(path))
    part = parsed.parts[0].flatten()
    assert part.getElementsByClass("KeySignature")[0].sharps == -3
    assert part.getElementsByClass("TimeSignature")[0].ratioString == "3/4"
    assert part.notes[0].pitch.name == "E-"


def test_empty_split_gives_one_measure_of_rest(tmp_path: Path) -> None:
    path = write_musicxml(build_score(HandSplit()), tmp_path / "vide.musicxml")
    parsed = converter.parse(str(path))
    assert len(parsed.parts[0].getElementsByClass("Measure")) == 1
    assert len(parsed.flatten().notes) == 0


def test_two_voices_and_triplets_round_trip(tmp_path: Path) -> None:
    notes = [q(0, 4, 60), q(1, 1, 67), q(2, F(1, 3), 69), q(2 + F(1, 3), F(1, 3), 71), q(2 + F(2, 3), F(1, 3), 72)]
    path = write_musicxml(build_score(HandSplit(right=notes)), tmp_path / "voix.musicxml")
    parsed = converter.parse(str(path))
    assert sorted(n.pitch.midi for n in parsed.parts[0].flatten().notes) == [60, 67, 69, 71, 72]


# --- rendus ------------------------------------------------------------------
def test_html_render(tmp_path: Path) -> None:
    musicxml = write_musicxml(build_score(split_hands(c_major_piece()), title="Gamme <1>"), tmp_path / "s.musicxml")
    pages = render_svg_pages(musicxml)
    assert len(pages) >= 1 and pages[0].lstrip().startswith("<svg")
    html_path = render_html(musicxml, tmp_path / "score.html", title="Gamme <1>")
    content = html_path.read_text(encoding="utf-8")
    assert "<svg" in content and "Gamme &lt;1&gt;" in content
    with pytest.raises(RenderError):
        render_svg_pages(tmp_path / "absent.musicxml")


def test_pdf_export_fallbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("partition.render.pdf_export.find_musescore", lambda configured=None: None)
    monkeypatch.setattr("partition.render.pdf_export.find_lilypond", lambda configured=None: None)
    musicxml = write_musicxml(build_score(split_hands(c_major_piece()), title="Gamme"), tmp_path / "s.musicxml")
    result = export_pdf(musicxml, tmp_path / "score.pdf", title="Gamme")
    assert result.method == "verovio"
    assert result.path is not None and result.path.read_bytes().startswith(b"%PDF")
    assert any("MuseScore" in warning for warning in result.warnings)

    nothing = export_pdf(musicxml, tmp_path / "none.pdf", allow_internal_fallback=False)
    assert nothing.path is None and nothing.method is None
    assert any("Aucun PDF" in warning for warning in nothing.warnings)


def test_pdf_export_reports_failing_musescore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setattr("partition.render.pdf_export.find_lilypond", lambda configured=None: None)
    musicxml = write_musicxml(build_score(split_hands(c_major_piece())), tmp_path / "s.musicxml")
    # Un « MuseScore » qui échoue : l'interpréteur Python appelé avec des arguments qu'il refuse.
    result = export_pdf(musicxml, tmp_path / "x.pdf", musescore_path=sys.executable, allow_internal_fallback=False)
    assert result.path is None
    assert any("MuseScore en échec" in warning for warning in result.warnings)


def test_find_executables_accept_missing_configuration() -> None:
    assert find_musescore("chemin/inexistant/mscore") in (None, find_musescore())
    assert find_lilypond("chemin/inexistant/lilypond") in (None, find_lilypond())


# --- MIDI --------------------------------------------------------------------
def test_midi_round_trip(tmp_path: Path) -> None:
    notes = [NoteEvent(0.0, 0.5, 60, 80), NoteEvent(0.5, 1.0, 64, 90, instrument="violin")]
    path = notes_to_midi(notes, tmp_path / "raw.mid")
    loaded = midi_to_notes(path)
    assert [(n.pitch, n.instrument) for n in loaded] == [(60, "piano"), (64, "violin")]
    assert loaded[0].offset == pytest.approx(0.5, abs=0.005)


def test_quantized_midi_round_trip(tmp_path: Path) -> None:
    notes = [q(0, 1, 60), q(1.25, 0.75, 64), q(2, F(1, 3), 67)]
    path = quantized_to_midi(notes, tmp_path / "quantized.mid", bpm=93.0)
    assert midi_tempo(path) == pytest.approx(93.0, abs=0.01)
    loaded, bpm = midi_to_quantized(path, grid=16, triplets=True)
    assert bpm == pytest.approx(93.0, abs=0.01)
    assert [(n.start, n.duration, n.pitch) for n in loaded] == [(n.start, n.duration, n.pitch) for n in notes]


def test_instrument_to_program() -> None:
    assert instrument_to_program("piano") == (0, False)
    assert instrument_to_program("violin") == (40, False)
    assert instrument_to_program("drums") == (0, True)
    assert instrument_to_program("program_12") == (12, False)
    assert instrument_to_program("inconnu") == (0, False)
