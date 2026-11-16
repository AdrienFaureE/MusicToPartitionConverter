"""Chargement audio : décodage, conversion mono, rééchantillonnage, normalisation.

Le décodage passe d'abord par ``soundfile`` (WAV, FLAC, OGG, MP3 avec libsndfile >= 1.1),
puis par ``ffmpeg`` pour les autres formats (M4A notamment).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a")


class AudioError(ValueError):
    """Fichier audio invalide, illisible ou trop long (message destiné à l'utilisateur)."""


@dataclass
class AudioInfo:
    """Résumé d'un fichier audio préparé."""

    path: Path
    sample_rate: int
    duration_s: float
    original_sample_rate: int
    decoder: str


def find_ffmpeg(configured: str | None = None) -> str | None:
    """Localise ffmpeg : chemin configuré, puis PATH, puis binaire fourni par ``imageio-ffmpeg``."""
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return str(candidate)
        found = shutil.which(configured)
        if found:
            return found
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # paquet absent ou binaire non fourni pour la plateforme
        return None


def _decode_soundfile(path: Path) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), int(sample_rate)


def _decode_ffmpeg(path: Path, ffmpeg: str, sample_rate: int) -> tuple[np.ndarray, int]:
    command = [
        ffmpeg, "-nostdin", "-v", "error", "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sample_rate), "-",
    ]  # fmt: skip
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except OSError as exc:
        raise AudioError(f"Impossible de lancer ffmpeg ({ffmpeg}) : {exc}") from exc
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace").strip().splitlines()
        raise AudioError(f"ffmpeg n'a pas pu décoder « {path.name} » : {detail[-1] if detail else 'erreur inconnue'}")
    return np.frombuffer(result.stdout, dtype=np.float32).copy(), sample_rate


def load_audio(
    path: str | Path,
    sample_rate: int = 22050,
    max_duration_s: float | None = None,
    normalize: bool = True,
    peak: float = 0.95,
    ffmpeg_path: str | None = None,
) -> tuple[np.ndarray, int, dict[str, object]]:
    """Charge un fichier audio en mono ``float32`` à la fréquence demandée.

    Returns:
        ``(échantillons, fréquence, métadonnées)`` ; les métadonnées contiennent
        ``original_sample_rate`` et ``decoder`` (``soundfile`` ou ``ffmpeg``).

    Raises:
        AudioError: fichier absent, format non pris en charge, décodage impossible,
            audio vide ou plus long que ``max_duration_s``.
    """
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"Fichier audio introuvable : {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioError(
            f"Format « {path.suffix or '?'} » non pris en charge. "
            f"Formats acceptés : {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    decoder = "soundfile"
    try:
        samples, original_sr = _decode_soundfile(path)
    except Exception as sf_error:
        ffmpeg = find_ffmpeg(ffmpeg_path)
        if ffmpeg is None:
            raise AudioError(
                f"Impossible de décoder « {path.name} » ({sf_error}). "
                "Installez ffmpeg (voir docs/prerequisites.md) pour lire ce format."
            ) from sf_error
        samples, original_sr = _decode_ffmpeg(path, ffmpeg, sample_rate)
        decoder = "ffmpeg"

    if samples.size == 0:
        raise AudioError(f"Le fichier « {path.name} » ne contient aucun échantillon audio.")
    duration = samples.size / original_sr
    if max_duration_s is not None and duration > max_duration_s:
        raise AudioError(
            f"Audio trop long : {duration:.0f} s pour un maximum de {max_duration_s:.0f} s "
            "(paramètre audio.max_duration_s)."
        )
    if original_sr != sample_rate:
        import librosa

        samples = librosa.resample(samples, orig_sr=original_sr, target_sr=sample_rate)
    samples = np.nan_to_num(samples.astype(np.float32))
    if normalize:
        maximum = float(np.max(np.abs(samples)))
        if maximum > 1e-9:
            samples = samples * (peak / maximum)
    return samples, sample_rate, {"original_sample_rate": original_sr, "decoder": decoder}


def prepare_audio(
    source: str | Path,
    destination: str | Path,
    sample_rate: int = 22050,
    max_duration_s: float | None = None,
    normalize: bool = True,
    peak: float = 0.95,
    ffmpeg_path: str | None = None,
) -> AudioInfo:
    """Convertit un fichier audio en WAV mono 16 bits prêt pour la transcription."""
    samples, sr, meta = load_audio(
        source, sample_rate=sample_rate, max_duration_s=max_duration_s,
        normalize=normalize, peak=peak, ffmpeg_path=ffmpeg_path,
    )  # fmt: skip
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), samples, sr, subtype="PCM_16")
    return AudioInfo(
        path=destination,
        sample_rate=sr,
        duration_s=samples.size / sr,
        original_sample_rate=int(meta["original_sample_rate"]),  # type: ignore[call-overload]
        decoder=str(meta["decoder"]),
    )
