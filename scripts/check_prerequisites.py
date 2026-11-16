"""Vérifie les prérequis : Python, ffmpeg, MuseScore, LilyPond, Git, paquets et moteurs.

Utilisation::

    python scripts/check_prerequisites.py

Code de sortie : 0 si tout l'indispensable est présent, 1 sinon. Les éléments optionnels
manquants sont signalés sans faire échouer le script.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OK, WARN, FAIL = "[ OK ]", "[ ?? ]", "[ KO ]"


def line(mark: str, label: str, detail: str = "") -> None:
    """Affiche une ligne de résultat."""
    print(f"{mark} {label:<28} {detail}")


def check_python() -> bool:
    """Python 3.10 ou 3.11 requis (3.12 : TensorFlow 2.15, épinglé par basic-pitch, n'est pas publié)."""
    version = sys.version_info
    text = f"{version.major}.{version.minor}.{version.micro}"
    if (version.major, version.minor) in ((3, 10), (3, 11)):
        line(OK, "Python", text)
        return True
    line(FAIL, "Python", f"{text} : utilisez Python 3.10 ou 3.11 (voir docs/prerequisites.md)")
    return False


def check_packages() -> bool:
    """Paquets Python indispensables au pipeline."""
    required = ["numpy", "librosa", "soundfile", "pretty_midi", "music21", "verovio", "fastapi", "uvicorn", "yaml"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        line(FAIL, "Paquets Python", f"absents : {', '.join(missing)} (pip install -r requirements.txt)")
        return False
    line(OK, "Paquets Python", "dépendances communes présentes")
    return True


def main() -> int:
    """Lance toutes les vérifications."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    print("Vérification des prérequis de partition-ia\n")
    essential = check_python()
    packages = check_packages()
    essential = essential and packages

    git = shutil.which("git")
    line(OK if git else WARN, "Git", git or "absent (utile seulement pour le dépôt et l'installation de MT3)")

    if packages:
        from partition.audio.load import find_ffmpeg
        from partition.config import load_settings
        from partition.render.pdf_export import find_lilypond, find_musescore
        from partition.transcription.registry import engine_statuses

        settings = load_settings()
        system_ffmpeg = shutil.which("ffmpeg")
        ffmpeg = find_ffmpeg(settings.paths.ffmpeg_path)
        if system_ffmpeg:
            line(OK, "ffmpeg", system_ffmpeg)
        elif ffmpeg:
            line(WARN, "ffmpeg", f"absent du PATH ; binaire de secours imageio-ffmpeg utilisé : {ffmpeg}")
        else:
            line(WARN, "ffmpeg", "absent : MP3/WAV/FLAC restent lisibles, pas le M4A")

        musescore = find_musescore(settings.paths.musescore_path)
        line(OK if musescore else WARN, "MuseScore (PDF)",
             musescore or "introuvable : PDF produit par le rendu de secours (définir MUSESCORE_PATH)")  # fmt: skip
        lilypond = find_lilypond(settings.paths.lilypond_path)
        line(OK if lilypond else WARN, "LilyPond (optionnel)", lilypond or "absent")

        print("\nMoteurs de transcription :")
        available = 0
        for status in engine_statuses(settings):
            available += status.available
            line(OK if status.available else WARN, status.name, status.detail)
        if available == 0:
            line(FAIL, "Moteurs", "aucun moteur disponible (python scripts/download_models.py)")
            essential = False

    try:
        import psutil  # type: ignore[import-untyped]

        ram = psutil.virtual_memory().total / 2**30
        line(OK if ram >= 15 else WARN, "Mémoire vive", f"{ram:.0f} Go (16 Go recommandés)")
    except ImportError:
        line(WARN, "Mémoire vive", "non mesurée (pip install psutil) ; 16 Go recommandés")
    try:
        import torch

        cuda = torch.cuda.is_available()
        line(OK, "GPU CUDA (optionnel)", "disponible" if cuda else "absent : calcul sur CPU")
    except ImportError:
        line(WARN, "GPU CUDA (optionnel)", "PyTorch absent")

    print("\nRésultat :", "prêt." if essential else "des éléments indispensables manquent.")
    return 0 if essential else 1


if __name__ == "__main__":
    raise SystemExit(main())
