"""Interface en ligne de commande.

Exemples::

    partition-ia transcribe morceau.mp3 --engine basic_pitch --output data/outputs/test
    partition-ia rebuild data/outputs/test --from quantized
    partition-ia engines
    partition-ia serve
"""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path

from partition import __version__
from partition.config import ConfigError, Settings, load_settings
from partition.logging_utils import setup_logging
from partition.pipeline import PipelineError, PipelineResult, run_pipeline
from partition.transcription.registry import available_engine_names, engine_statuses


def build_parser() -> argparse.ArgumentParser:
    """Construit l'analyseur d'arguments."""
    parser = argparse.ArgumentParser(prog="partition-ia", description="Transcription audio piano vers partition.")
    parser.add_argument("--version", action="version", version=f"partition-ia {__version__}")
    parser.add_argument("--config", help="Fichier YAML fusionné par-dessus config/default.yaml.")
    commands = parser.add_subparsers(dest="command", required=True)

    transcribe = commands.add_parser("transcribe", help="Transcrire un fichier audio.")
    transcribe.add_argument("audio", help="Fichier audio (mp3, wav, flac, m4a).")
    transcribe.add_argument("--engine", help="bytedance, mt3, basic_pitch ou all (défaut : configuration).")
    transcribe.add_argument("--output", help="Dossier de sortie (défaut : data/outputs/<identifiant>).")
    transcribe.add_argument("--hand-overrides", help="Fichier JSON d'override de la séparation des mains.")
    transcribe.add_argument("--title", help="Titre de la partition.")

    rebuild = commands.add_parser("rebuild", help="Reconstruire la partition d'un job existant sans retranscrire.")
    rebuild.add_argument("job_dir", help="Dossier d'un job (contenant raw.mid ou quantized.mid).")
    rebuild.add_argument("--from", dest="start_from", choices=("raw", "quantized"), default="raw",
                         help="raw : reprend au nettoyage ; quantized : reprend à la tonalité.")  # fmt: skip
    rebuild.add_argument("--hand-overrides", help="Fichier JSON d'override de la séparation des mains.")
    rebuild.add_argument("--title", help="Titre de la partition.")

    commands.add_parser("engines", help="Lister les moteurs et leur disponibilité.")

    serve = commands.add_parser("serve", help="Lancer l'API et l'interface web locales.")
    serve.add_argument("--host", help="Adresse d'écoute (défaut : configuration).")
    serve.add_argument("--port", type=int, help="Port (défaut : configuration).")
    return parser


def _print_result(result: PipelineResult) -> None:
    report = result.report
    tempo = report.get("tempo", {})
    key = report.get("key", {})
    print(f"[{result.engine}] Terminé en {report.get('total_s', 0):.1f} s → {result.output_dir}")
    print(f"  Tempo : {tempo.get('bpm')} ({tempo.get('method')}) | Tonalité : {key.get('name')} "
          f"| Mesures : {report.get('measures')} | Notes : {report.get('notes')}")  # fmt: skip
    print(f"  Fichiers : {', '.join(sorted(result.files))}")
    for warning in report.get("warnings", []):
        print(f"  Avertissement : {warning}")


def _cmd_transcribe(args: argparse.Namespace, settings: Settings) -> int:
    engine = args.engine or settings.engine
    engines = available_engine_names() if engine == "all" else [engine]
    base = Path(args.output) if args.output else settings.outputs_dir / uuid.uuid4().hex[:12]
    failures = 0
    for name in engines:
        output_dir = base / name if len(engines) > 1 else base
        try:
            result = run_pipeline(
                output_dir, engine=name, audio_path=args.audio, settings=settings,
                overrides_path=args.hand_overrides, title=args.title,
            )  # fmt: skip
            _print_result(result)
        except PipelineError as exc:
            failures += 1
            print(f"[{name}] Erreur : {exc}", file=sys.stderr)
    return 1 if failures == len(engines) else 0


def _cmd_rebuild(args: argparse.Namespace, settings: Settings) -> int:
    try:
        result = run_pipeline(
            args.job_dir, settings=settings, start_from=args.start_from,
            overrides_path=args.hand_overrides, title=args.title,
        )  # fmt: skip
    except PipelineError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    _print_result(result)
    return 0


def _cmd_engines(settings: Settings) -> int:
    for status in engine_statuses(settings):
        mark = "disponible" if status.available else "indisponible"
        multi = "multi-instrument" if status.supports_multi_instrument else "piano"
        print(f"{status.name:<12} {mark:<13} {multi:<17} {status.detail}")
    return 0


def _cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    try:
        import uvicorn
    except ImportError:
        print("Erreur : uvicorn est absent (pip install -r requirements.txt).", file=sys.stderr)
        return 1
    uvicorn.run("api.main:app", host=args.host or settings.server.host, port=args.port or settings.server.port)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée de la commande ``partition-ia``. Retourne le code de sortie."""
    for stream in (sys.stdout, sys.stderr):  # consoles Windows en cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    setup_logging()
    try:
        settings = load_settings(args.config)
    except ConfigError as exc:
        print(f"Erreur de configuration : {exc}", file=sys.stderr)
        return 2
    if args.command == "transcribe":
        return _cmd_transcribe(args, settings)
    if args.command == "rebuild":
        return _cmd_rebuild(args, settings)
    if args.command == "engines":
        return _cmd_engines(settings)
    return _cmd_serve(args, settings)


if __name__ == "__main__":
    raise SystemExit(main())
