"""Tests du chargement audio."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from partition.audio.load import AudioError, find_ffmpeg, load_audio, prepare_audio


def test_load_wav_mono(fixture_files: dict[str, Path]) -> None:
    samples, sample_rate, meta = load_audio(fixture_files["piano_wav"], sample_rate=22050)
    assert sample_rate == 22050
    assert samples.ndim == 1 and samples.dtype == np.float32
    assert 9.5 < len(samples) / sample_rate < 11.5
    assert float(np.max(np.abs(samples))) == pytest.approx(0.95, abs=0.01)
    assert meta["decoder"] == "soundfile"


def test_stereo_is_downmixed_and_resampled(fixture_files: dict[str, Path]) -> None:
    samples, sample_rate, meta = load_audio(fixture_files["stereo_wav"], sample_rate=16000, normalize=False)
    assert sample_rate == 16000
    assert meta["original_sample_rate"] == 44100
    assert abs(len(samples) - 16000) <= 2
    assert float(np.max(np.abs(samples))) < 0.5  # moyenne des deux canaux


def test_load_flac(fixture_files: dict[str, Path]) -> None:
    samples, _, _ = load_audio(fixture_files["piano_flac"])
    assert samples.size > 0


def test_too_long_is_rejected(fixture_files: dict[str, Path]) -> None:
    with pytest.raises(AudioError, match="trop long"):
        load_audio(fixture_files["piano_wav"], max_duration_s=5)


def test_missing_file() -> None:
    with pytest.raises(AudioError, match="introuvable"):
        load_audio("absent.wav")


def test_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("pas de l'audio", encoding="utf-8")
    with pytest.raises(AudioError, match="non pris en charge"):
        load_audio(path)


def test_corrupted_file(tmp_path: Path) -> None:
    path = tmp_path / "casse.wav"
    path.write_bytes(b"ceci n'est pas un fichier wav")
    with pytest.raises(AudioError):
        load_audio(path)


def test_prepare_audio_writes_mono_wav(fixture_files: dict[str, Path], tmp_path: Path) -> None:
    info = prepare_audio(fixture_files["stereo_wav"], tmp_path / "out" / "audio.wav", sample_rate=22050)
    assert info.path.is_file()
    assert info.sample_rate == 22050
    assert info.duration_s == pytest.approx(1.0, abs=0.01)
    data, sample_rate = sf.read(str(info.path), always_2d=True)
    assert sample_rate == 22050 and data.shape[1] == 1


def test_find_ffmpeg_returns_path_or_none() -> None:
    found = find_ffmpeg()
    assert found is None or Path(found).name.lower().startswith("ffmpeg")
    assert find_ffmpeg("executable_qui_n_existe_pas") == found


def test_m4a_is_decoded_through_ffmpeg(fixture_files: dict[str, Path], tmp_path: Path) -> None:
    import subprocess

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        pytest.skip("ffmpeg indisponible (ni dans le PATH, ni via imageio-ffmpeg)")
    target = tmp_path / "piano.m4a"
    subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", str(fixture_files["piano_wav"]), "-c:a", "aac", str(target)], check=True
    )
    samples, sample_rate, meta = load_audio(target, sample_rate=16000)
    assert meta["decoder"] == "ffmpeg" and sample_rate == 16000
    assert 9.5 < len(samples) / sample_rate < 11.5
