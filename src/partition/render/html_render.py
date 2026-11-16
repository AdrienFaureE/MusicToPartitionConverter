"""Rendu MusicXML → SVG (Verovio) assemblé dans un ``score.html`` autonome."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any


class RenderError(RuntimeError):
    """Le rendu de la partition a échoué."""


DEFAULT_OPTIONS: dict[str, Any] = {
    "pageWidth": 2100,  # A4 en dixièmes de millimètre
    "pageHeight": 2970,
    "pageMarginTop": 100,
    "pageMarginBottom": 100,
    "pageMarginLeft": 100,
    "pageMarginRight": 100,
    "scale": 40,
    "footer": "none",
    "adjustPageHeight": False,
    "breaks": "auto",
}

_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ margin: 0; background: #eceef1; font-family: system-ui, sans-serif; color: #1c2430; }}
  header {{ padding: 12px 20px; font-size: 14px; color: #4a5565; }}
  .page {{ background: #fff; margin: 0 auto 16px; max-width: 840px; box-shadow: 0 1px 4px rgba(0,0,0,.18); }}
  .page svg {{ display: block; width: 100%; height: auto; }}
  @media print {{
    body {{ background: #fff; }}
    header {{ display: none; }}
    .page {{ box-shadow: none; margin: 0; max-width: none; page-break-after: always; }}
  }}
</style>
</head>
<body>
<header>{title} — {count} page(s)</header>
{pages}
</body>
</html>
"""


def render_svg_pages(musicxml_path: str | Path, options: dict[str, Any] | None = None) -> list[str]:
    """Rend un fichier MusicXML en une liste de SVG (un par page).

    Raises:
        RenderError: Verovio absent ou fichier illisible.
    """
    try:
        import verovio
    except ImportError as exc:
        raise RenderError("Le paquet Python « verovio » est absent (pip install verovio).") from exc
    musicxml_path = Path(musicxml_path)
    if not musicxml_path.is_file():
        raise RenderError(f"Fichier MusicXML introuvable : {musicxml_path}")
    # Le chemin des polices par défaut n'est pas visible depuis un thread secondaire (jobs de l'API) :
    # on le fixe explicitement sur chaque instance.
    toolkit = verovio.toolkit(False)
    resources = Path(verovio.__file__).resolve().parent / "data"
    loaded = toolkit.setResourcePath(str(resources))  # type: ignore[func-returns-value]
    if loaded is False:  # le binaire retourne un booléen, contrairement à ce qu'annonce son fichier .pyi
        raise RenderError(f"Ressources Verovio introuvables : {resources}")
    toolkit.setOptions({**DEFAULT_OPTIONS, **(options or {})})
    if not toolkit.loadData(musicxml_path.read_text(encoding="utf-8")):
        raise RenderError(f"Verovio n'a pas pu lire {musicxml_path.name}.")
    return [toolkit.renderToSVG(page) for page in range(1, toolkit.getPageCount() + 1)]


def render_html(
    musicxml_path: str | Path,
    html_path: str | Path,
    title: str = "Partition",
    options: dict[str, Any] | None = None,
) -> Path:
    """Écrit un ``score.html`` autonome contenant toutes les pages SVG. Retourne son chemin."""
    pages = render_svg_pages(musicxml_path, options)
    body = "\n".join(f'<div class="page">{svg}</div>' for svg in pages)
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(
        _TEMPLATE.format(title=html.escape(title), count=len(pages), pages=body), encoding="utf-8"
    )
    return html_path
