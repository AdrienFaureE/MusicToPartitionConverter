"""Export MusicXML → PDF.

Ordre des méthodes :

1. MuseScore en ligne de commande (``mscore -o score.pdf score.musicxml``), qualité de référence ;
2. LilyPond (``musicxml2ly`` puis ``lilypond``) ;
3. secours interne : pages SVG de Verovio converties en PDF par ``svglib``/``reportlab``
   (aucun logiciel externe, gravure plus simple, certains glyphes de texte peuvent manquer).

Si aucune méthode n'aboutit, la fonction ne lève pas d'erreur : elle retourne un résultat sans
chemin, avec des avertissements à reporter dans ``report.json``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

MUSESCORE_NAMES = ("mscore", "mscore4", "musescore", "musescore4", "MuseScore4", "mscore4portable", "mscore3", "MuseScore3")


@dataclass
class PdfResult:
    """Résultat de l'export PDF."""

    path: Path | None = None
    method: str | None = None
    warnings: list[str] = field(default_factory=list)


def _existing(candidate: str | Path | None) -> str | None:
    if not candidate:
        return None
    path = Path(candidate).expanduser()
    if path.is_file():
        return str(path)
    return shutil.which(str(candidate))


def musescore_candidates() -> list[Path]:
    """Emplacements d'installation habituels de MuseScore selon le système."""
    candidates: list[Path] = []
    if sys.platform == "win32":
        for variable in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)", "LOCALAPPDATA"):
            base = os.environ.get(variable)
            if not base:
                continue
            for folder, executable in (
                ("MuseScore 4", "MuseScore4.exe"),
                ("MuseScore Studio 4", "MuseScore4.exe"),
                ("MuseScore 3", "MuseScore3.exe"),
            ):
                candidates.append(Path(base) / folder / "bin" / executable)
                candidates.append(Path(base) / "Programs" / folder / "bin" / executable)
    elif sys.platform == "darwin":
        for root in (Path("/Applications"), Path.home() / "Applications"):
            for app in ("MuseScore 4.app", "MuseScore Studio.app", "MuseScore 3.app"):
                candidates.append(root / app / "Contents" / "MacOS" / "mscore")
    else:
        candidates += [Path("/usr/bin/mscore"), Path("/usr/local/bin/mscore"), Path("/snap/bin/musescore")]
        candidates += sorted(Path.home().glob("*.AppImage"))
    return candidates


def find_musescore(configured: str | None = None) -> str | None:
    """Localise l'exécutable MuseScore : chemin configuré, PATH, puis emplacements habituels."""
    found = _existing(configured)
    if found:
        return found
    for name in MUSESCORE_NAMES:
        found = shutil.which(name)
        if found:
            return found
    for candidate in musescore_candidates():
        if candidate.is_file() and ("musescore" in candidate.name.lower() or "mscore" in candidate.name.lower()):
            return str(candidate)
    return None


def find_lilypond(configured: str | None = None) -> str | None:
    """Localise l'exécutable LilyPond : chemin configuré, puis PATH."""
    return _existing(configured) or shutil.which("lilypond")


def _run(command: list[str], timeout: int, cwd: Path | None = None) -> None:
    env = dict(os.environ)
    if sys.platform.startswith("linux") and not env.get("DISPLAY"):
        env.setdefault("QT_QPA_PLATFORM", "offscreen")  # MuseScore sans écran
    result = subprocess.run(
        command, capture_output=True, text=True, errors="replace", timeout=timeout, cwd=cwd, env=env, check=False
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(f"code de retour {result.returncode} : {' | '.join(tail) or 'aucun message'}")


def export_with_musescore(musicxml: Path, pdf: Path, executable: str, timeout: int = 300) -> Path:
    """Convertit avec MuseScore (``mscore -o sortie.pdf entrée.musicxml``)."""
    _run([executable, "-o", str(pdf), str(musicxml)], timeout)
    if not pdf.is_file() or pdf.stat().st_size == 0:
        raise RuntimeError("MuseScore n'a produit aucun PDF.")
    return pdf


def _musicxml2ly_command(lilypond: str) -> list[str] | None:
    """Commande ``musicxml2ly`` : exécutable du PATH, ou script voisin de LilyPond lancé par son Python."""
    found = shutil.which("musicxml2ly")
    if found:
        return [found]
    bin_dir = Path(lilypond).resolve().parent
    for name in ("musicxml2ly", "musicxml2ly.py"):
        script = bin_dir / name
        if script.is_file():
            # TODO: vérifier sur une installation LilyPond Windows (le script y est lancé par le Python fourni).
            bundled = next((p for p in (bin_dir / "python.exe", bin_dir / "python3", bin_dir / "python") if p.is_file()), None)
            return [str(bundled or sys.executable), str(script)]
    return None


def export_with_lilypond(musicxml: Path, pdf: Path, executable: str, timeout: int = 300) -> Path:
    """Convertit avec LilyPond : ``musicxml2ly`` puis ``lilypond``."""
    converter = _musicxml2ly_command(executable)
    if converter is None:
        raise RuntimeError("musicxml2ly introuvable à côté de LilyPond.")
    with tempfile.TemporaryDirectory() as tmp:
        ly_file = Path(tmp) / "score.ly"
        _run([*converter, "-o", str(ly_file), str(musicxml)], timeout)
        _run([executable, "-o", str(Path(tmp) / "score"), str(ly_file)], timeout, cwd=Path(tmp))
        produced = Path(tmp) / "score.pdf"
        if not produced.is_file():
            raise RuntimeError("LilyPond n'a produit aucun PDF.")
        shutil.copyfile(produced, pdf)
    return pdf


def export_with_verovio(musicxml: Path, pdf: Path, title: str | None = None, author: str | None = None) -> Path:
    """Secours interne : pages SVG Verovio converties en PDF vectoriel (A4)."""
    from reportlab.graphics import renderPDF
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from svglib.svglib import svg2rlg

    from partition.render.html_render import render_svg_pages

    pages = render_svg_pages(musicxml, {"header": "none", "pageMarginTop": 200})
    width, height = A4
    document = canvas.Canvas(str(pdf), pagesize=A4)
    document.setTitle(title or "Partition")
    document.setAuthor(author or "")
    document.setCreator("partition-ia")
    with tempfile.TemporaryDirectory() as tmp:
        for index, svg in enumerate(pages):
            svg_file = Path(tmp) / f"page{index}.svg"
            svg_file.write_text(svg, encoding="utf-8")
            drawing = svg2rlg(str(svg_file))
            if drawing is None:
                raise RuntimeError("Conversion SVG impossible.")
            factor = min(width / drawing.width, height / drawing.height)
            drawing.scale(factor, factor)
            renderPDF.draw(drawing, document, 0, height - drawing.height * factor)
            if index == 0 and title:
                document.setFont("Times-Roman", 18)
                document.drawCentredString(width / 2, height - 42, title)
            document.showPage()
    document.save()
    return pdf


def export_pdf(
    musicxml_path: str | Path,
    pdf_path: str | Path,
    musescore_path: str | None = None,
    lilypond_path: str | None = None,
    title: str | None = None,
    author: str | None = None,
    allow_internal_fallback: bool = True,
) -> PdfResult:
    """Exporte le PDF avec la première méthode disponible (voir l'en-tête du module)."""
    musicxml = Path(musicxml_path)
    pdf = Path(pdf_path)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    result = PdfResult()

    musescore = find_musescore(musescore_path)
    if musescore:
        try:
            result.path, result.method = export_with_musescore(musicxml, pdf, musescore), "musescore"
            return result
        except Exception as exc:
            result.warnings.append(f"Export PDF par MuseScore en échec ({exc}).")
    else:
        result.warnings.append("MuseScore introuvable (définir MUSESCORE_PATH ou paths.musescore_path).")

    lilypond = find_lilypond(lilypond_path)
    if lilypond:
        try:
            result.path, result.method = export_with_lilypond(musicxml, pdf, lilypond), "lilypond"
            return result
        except Exception as exc:
            result.warnings.append(f"Export PDF par LilyPond en échec ({exc}).")
    else:
        result.warnings.append("LilyPond introuvable.")

    if allow_internal_fallback:
        try:
            result.path, result.method = export_with_verovio(musicxml, pdf, title, author), "verovio"
            result.warnings.append(
                "PDF produit par le rendu de secours Verovio : installez MuseScore 4 pour une gravure de référence."
            )
            return result
        except Exception as exc:
            result.warnings.append(f"Rendu PDF de secours en échec ({exc}).")
    result.warnings.append("Aucun PDF produit : ouvrez score.musicxml dans MuseScore, ou imprimez score.html en PDF.")
    return result
