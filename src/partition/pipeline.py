"""Orchestration complète d'un job : de l'audio à la partition.

Étapes (chacune journalisée dans ``job.log`` et sauvegardée dans le dossier de sortie) :

1. chargement audio → ``audio.wav`` ;
2. transcription → ``raw.mid``, ``raw_notes.json`` ;
3. nettoyage → ``cleaned.mid`` ;
4. tempo et grille ;
5. quantification → ``quantized.mid`` ;
6. tonalité et métrique ;
7. séparation des mains (override possible par ``hand_overrides.json``) ;
8. construction music21 ;
9. exports → ``score.musicxml``, ``score.pdf``, ``score.html``, ``report.json``.

Le pipeline peut repartir de ``raw.mid`` (``start_from="raw"``) ou de ``quantized.mid``
(``start_from="quantized"``) pour comparer les moteurs sur la même chaîne aval.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from partition.audio.load import AudioError, prepare_audio
from partition.config import Settings, load_settings
from partition.logging_utils import close_job_logger, get_job_logger
from partition.midi_io import midi_to_notes, midi_to_quantized, notes_to_midi, quantized_to_midi
from partition.notation.key_meter import key_label, measure_length, parse_time_signature, resolve_key
from partition.notation.musicxml_builder import build_score, write_musicxml
from partition.postprocess.cleanup import clean_notes
from partition.postprocess.hand_split import OVERRIDES_FILENAME, HandOverrideError, load_overrides, split_hands
from partition.postprocess.quantize import quantize_notes
from partition.postprocess.tempo import estimate_grid
from partition.render.html_render import render_html
from partition.render.pdf_export import export_pdf
from partition.transcription.base import PIANO, EngineUnavailableError, NoteEvent, TranscriptionError
from partition.transcription.registry import get_engine

START_POINTS = ("audio", "raw", "quantized")
ProgressCallback = Callable[[str, float, str], None]

# Étape → (libellé, progression atteinte à la fin de l'étape)
STEPS: dict[str, tuple[str, float]] = {
    "audio": ("Chargement audio", 0.05),
    "transcription": ("Transcription", 0.70),
    "cleanup": ("Nettoyage", 0.74),
    "tempo": ("Tempo et grille", 0.80),
    "quantize": ("Quantification", 0.83),
    "key_meter": ("Tonalité et métrique", 0.85),
    "hands": ("Séparation des mains", 0.87),
    "notation": ("Construction de la partition", 0.90),
    "exports": ("Exports", 1.0),
}


class PipelineError(RuntimeError):
    """Erreur du pipeline, avec un message lisible par l'utilisateur."""


@dataclass
class PipelineResult:
    """Résultat d'un job."""

    output_dir: Path
    engine: str
    report: dict[str, Any]
    files: dict[str, Path] = field(default_factory=dict)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(
    output_dir: str | Path,
    engine: str | None = None,
    audio_path: str | Path | None = None,
    settings: Settings | None = None,
    start_from: str = "audio",
    overrides_path: str | Path | None = None,
    progress: ProgressCallback | None = None,
    job_id: str | None = None,
    title: str | None = None,
) -> PipelineResult:
    """Exécute le pipeline et retourne les fichiers produits.

    Args:
        output_dir: dossier de sortie du job (créé si besoin).
        engine: nom du moteur ; par défaut celui de la configuration.
        audio_path: fichier audio source (obligatoire si ``start_from="audio"``).
        settings: configuration ; chargée par défaut depuis le YAML et l'environnement.
        start_from: ``audio`` (complet), ``raw`` (depuis ``raw.mid``) ou ``quantized`` (depuis ``quantized.mid``).
        overrides_path: fichier JSON d'override des mains ; sinon ``hand_overrides.json`` du dossier.
        progress: rappel ``(étape, fraction 0-1, message)``.
        job_id: identifiant utilisé pour le journal.
        title: titre de la partition (sinon configuration, sinon nom du fichier audio).

    Raises:
        PipelineError: erreur explicable à l'utilisateur (fichier invalide, moteur indisponible...).
    """
    settings = settings or load_settings()
    if start_from not in START_POINTS:
        raise PipelineError(f"Point de reprise inconnu : « {start_from} » (attendu : {', '.join(START_POINTS)}).")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = get_job_logger(out, job_id)
    previous_report: dict[str, Any] = {}
    if start_from != "audio" and (out / "report.json").is_file():
        try:
            previous_report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_report = {}
    engine_name = engine or previous_report.get("engine") or settings.engine

    report: dict[str, Any] = {
        "status": "running",
        "engine": engine_name,
        "start_from": start_from,
        "started_at": _now(),
        "source": str(audio_path) if audio_path else previous_report.get("source"),
        "audio": previous_report.get("audio"),
        "timings_s": {},
        "notes": {},
        "warnings": [],
        "files": [],
    }
    warnings: list[str] = report["warnings"]
    timings: dict[str, float] = report["timings_s"]
    files: dict[str, Path] = {}
    step_start = time.perf_counter()
    total_start = step_start

    def begin(step: str) -> None:
        nonlocal step_start
        step_start = time.perf_counter()
        label = STEPS[step][0]
        logger.info("Étape « %s »...", label)
        if progress:
            previous = [p for name, (_, p) in STEPS.items() if list(STEPS).index(name) < list(STEPS).index(step)]
            progress(step, previous[-1] if previous else 0.0, label)

    def end(step: str, message: str = "") -> None:
        timings[step] = round(time.perf_counter() - step_start, 3)
        logger.info("Étape « %s » terminée en %.2f s. %s", STEPS[step][0], timings[step], message)
        if progress:
            progress(step, STEPS[step][1], STEPS[step][0])

    def warn(message: str) -> None:
        warnings.append(message)
        logger.warning(message)

    try:
        wav_path = out / "audio.wav"
        raw_path = out / "raw.mid"
        quantized_path = out / "quantized.mid"
        quantized = []
        bpm = 120.0
        tempo_method = "midi-file"

        if start_from == "audio":
            # 1. Chargement audio
            begin("audio")
            if audio_path is None:
                raise PipelineError("Aucun fichier audio fourni.")
            info = prepare_audio(
                audio_path, wav_path,
                sample_rate=settings.audio.sample_rate, max_duration_s=settings.audio.max_duration_s,
                normalize=settings.audio.normalize, peak=settings.audio.peak,
                ffmpeg_path=settings.paths.ffmpeg_path,
            )  # fmt: skip
            report["audio"] = {
                "duration_s": round(info.duration_s, 2),
                "sample_rate": info.sample_rate,
                "original_sample_rate": info.original_sample_rate,
                "decoder": info.decoder,
            }
            end("audio", f"Durée : {info.duration_s:.1f} s.")

            # 2. Transcription
            begin("transcription")
            model = get_engine(engine_name, settings)
            model.load()
            load_time = time.perf_counter() - step_start
            raw_notes = model.transcribe(wav_path)
            notes_to_midi(raw_notes, raw_path)
            _write_json(out / "raw_notes.json", [n.to_dict() for n in raw_notes])
            end("transcription", f"{len(raw_notes)} notes.")
            report["engine_load_s"] = round(load_time, 3)
            transcribe_time = timings["transcription"] - load_time
            report["transcription_s_per_audio_minute"] = round(transcribe_time / max(info.duration_s, 1e-6) * 60, 2)
            backend = getattr(model, "backend", None)
            if backend:
                report["engine_backend"] = backend
        elif start_from == "raw":
            if not raw_path.is_file():
                raise PipelineError(f"Reprise impossible : {raw_path} est absent.")
            raw_notes = midi_to_notes(raw_path)
            logger.info("Reprise depuis %s (%d notes).", raw_path.name, len(raw_notes))
        else:
            if not quantized_path.is_file():
                raise PipelineError(f"Reprise impossible : {quantized_path} est absent.")
            raw_notes = []

        if start_from in ("audio", "raw"):
            report["notes"]["raw"] = len(raw_notes)
            instruments = sorted({n.instrument for n in raw_notes})
            report["instruments"] = instruments

            # 3. Nettoyage
            begin("cleanup")
            cleaned, stats = clean_notes(
                raw_notes, settings.cleanup.min_duration_s, settings.cleanup.min_velocity, settings.cleanup.merge_window_s
            )
            report["cleanup"] = stats
            report["notes"]["cleaned"] = len(cleaned)
            if not cleaned:
                raise PipelineError("Aucune note détectée dans l'audio après nettoyage.")
            notes_to_midi(cleaned, out / "cleaned.mid")
            end("cleanup", f"{len(cleaned)} notes conservées sur {len(raw_notes)}.")

            # 4. Tempo et grille
            begin("tempo")
            piano_notes: list[NoteEvent] = [n for n in cleaned if n.instrument == PIANO] or cleaned
            grid, tempo_warnings = estimate_grid(
                piano_notes,
                audio_path=wav_path if wav_path.is_file() else None,
                method=settings.tempo.method, forced_bpm=settings.tempo.forced_bpm,
                min_bpm=settings.tempo.min_bpm, max_bpm=settings.tempo.max_bpm,
                follow_beats=settings.tempo.follow_beats,
            )  # fmt: skip
            for message in tempo_warnings:
                warn(message)
            bpm, tempo_method = grid.bpm, grid.method
            end("tempo", f"{bpm:.1f} à la noire ({tempo_method}).")

            # 5. Quantification
            begin("quantize")
            quantized = quantize_notes(
                cleaned, grid, settings.quantize.grid, settings.quantize.triplets, settings.quantize.fill_gap_beats
            )
            quantized_to_midi(quantized, quantized_path, bpm)
            end("quantize", f"Grille : 1/{settings.quantize.grid}.")
        else:
            quantized, bpm = midi_to_quantized(quantized_path, settings.quantize.grid, settings.quantize.triplets)
            logger.info("Reprise depuis %s (%d notes, %.1f à la noire).", quantized_path.name, len(quantized), bpm)
            if not quantized:
                raise PipelineError("quantized.mid ne contient aucune note.")
        report["notes"]["quantized"] = len(quantized)
        report["tempo"] = {"bpm": round(bpm, 2), "method": tempo_method}

        # 6. Tonalité et métrique
        begin("key_meter")
        piano_quantized = [n for n in quantized if n.instrument == PIANO]
        try:
            key, key_source = resolve_key(piano_quantized, settings.notation.key)
            numerator, denominator = parse_time_signature(settings.notation.time_signature)
        except ValueError as exc:
            raise PipelineError(str(exc)) from exc
        time_signature = f"{numerator}/{denominator}"
        report["key"] = {"name": key_label(key), "sharps": key.sharps, "source": key_source}
        report["time_signature"] = time_signature
        end("key_meter", f"{key_label(key)}, {time_signature}.")

        # 7. Séparation des mains
        begin("hands")
        override_file = Path(overrides_path) if overrides_path else out / OVERRIDES_FILENAME
        overrides = None
        if override_file.is_file():
            overrides = load_overrides(override_file)
            if override_file.resolve() != (out / OVERRIDES_FILENAME).resolve():
                shutil.copyfile(override_file, out / OVERRIDES_FILENAME)
            report["hand_overrides"] = True
        elif overrides_path:
            raise PipelineError(f"Fichier d'override introuvable : {override_file}")
        split = split_hands(
            quantized, settings.hands.method, settings.hands.split_pitch, settings.hands.max_hand_span,
            overrides=overrides, beats_per_measure=float(measure_length(time_signature)),
        )  # fmt: skip
        report["notes"].update(right=len(split.right), left=len(split.left), others=len(split.others))
        if split.others:
            quantized_to_midi(split.others, out / "others.mid", bpm)
            files["others.mid"] = out / "others.mid"
            names = ", ".join(sorted({n.instrument for n in split.others}))
            warn(f"Instruments non notés en V1 ({names}) : {len(split.others)} notes enregistrées dans others.mid.")
        if not split.right and not split.left:
            raise PipelineError("Aucune note de piano à noter.")
        end("hands", f"Main droite : {len(split.right)} notes, main gauche : {len(split.left)} notes.")

        # 8. Construction music21
        begin("notation")
        source_name = Path(str(report["source"])).stem if report.get("source") else None
        score_title = title or settings.notation.title or source_name or "Partition"
        score = build_score(
            split, key, time_signature, bpm=bpm, title=score_title,
            composer=settings.notation.composer, max_voices_per_staff=settings.notation.max_voices_per_staff,
            author=settings.notation.author,
        )  # fmt: skip
        report["measures"] = len(score.parts[0].getElementsByClass("Measure"))
        report["title"] = score_title
        end("notation", f"{report['measures']} mesures.")

        # 9. Exports
        begin("exports")
        musicxml_path = write_musicxml(score, out / "score.musicxml")
        files["score.musicxml"] = musicxml_path
        try:
            files["score.html"] = render_html(musicxml_path, out / "score.html", title=score_title)
        except Exception as exc:
            warn(f"Rendu HTML impossible : {exc}")
        pdf = export_pdf(
            musicxml_path, out / "score.pdf",
            musescore_path=settings.paths.musescore_path, lilypond_path=settings.paths.lilypond_path,
            title=score_title, author=settings.notation.author,
        )  # fmt: skip
        report["pdf_method"] = pdf.method
        if pdf.path:
            files["score.pdf"] = pdf.path
            for message in pdf.warnings:
                logger.info(message)
            if pdf.method == "verovio":
                warn(pdf.warnings[-1])
        else:
            for message in pdf.warnings:
                warn(message)
        for name in ("raw.mid", "cleaned.mid", "quantized.mid", "audio.wav", "raw_notes.json"):
            if (out / name).is_file():
                files[name] = out / name
        end("exports")

        report["status"] = "done"
    except (PipelineError, AudioError, EngineUnavailableError, TranscriptionError, HandOverrideError) as exc:
        report["status"] = "error"
        report["error"] = str(exc)
        logger.error("Échec du job : %s", exc)
        raise PipelineError(str(exc)) from exc
    except Exception as exc:
        report["status"] = "error"
        report["error"] = f"Erreur inattendue : {exc}"
        logger.exception("Erreur inattendue pendant le job")
        raise PipelineError(f"Erreur inattendue pendant le traitement ({type(exc).__name__}). Voir job.log.") from exc
    finally:
        report["finished_at"] = _now()
        report["total_s"] = round(time.perf_counter() - total_start, 3)
        files["report.json"] = out / "report.json"
        files["job.log"] = out / "job.log"
        report["files"] = sorted(files)
        _write_json(out / "report.json", report)
        close_job_logger(logger)

    return PipelineResult(output_dir=out, engine=engine_name, report=report, files=files)
