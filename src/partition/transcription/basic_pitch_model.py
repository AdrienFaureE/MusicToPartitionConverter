"""Moteur Basic Pitch (Spotify, paquet ``basic-pitch``).

API vérifiée dans le code de la version 0.4.0 :

- ``basic_pitch.inference.predict(audio_path, model_or_model_path, onset_threshold, frame_threshold,
  minimum_note_length, ...)`` retourne ``(model_output, midi_data, note_events)`` ;
- ``note_events`` est une liste de tuples ``(début_s, fin_s, hauteur_midi, amplitude 0-1, pitch_bends)`` ;
- ``basic_pitch.ICASSP_2022_MODEL_PATH`` désigne le premier backend disponible dans l'ordre
  TensorFlow, CoreML, TFLite, ONNX. Les poids sont livrés dans le paquet pip : rien à télécharger.

Basic Pitch est généraliste (toute source polyphonique) et ne distingue pas les instruments :
toutes les notes sont étiquetées « piano ».
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from partition.transcription.base import (
    PIANO,
    EngineUnavailableError,
    NoteEvent,
    TranscriptionError,
    TranscriptionModel,
    normalize_events,
)


class BasicPitchModel(TranscriptionModel):
    """Transcription polyphonique légère."""

    name = "basic_pitch"
    supports_multi_instrument = False

    def __init__(self, options: dict[str, Any] | None = None, models_dir: Path | None = None) -> None:
        super().__init__(options, models_dir)
        self._model: Any = None
        self.backend: str | None = None

    def check_available(self) -> str | None:
        if importlib.util.find_spec("basic_pitch") is None:
            return "Paquet Python « basic-pitch » absent (pip install -r requirements.txt)."
        backends = ("tensorflow", "coremltools", "tflite_runtime", "onnxruntime")
        if not any(importlib.util.find_spec(name) is not None for name in backends):
            return "Aucun backend d'inférence installé (tensorflow, onnxruntime, tflite-runtime ou coremltools)."
        return None

    def load(self) -> None:
        if self._loaded:
            return
        reason = self.check_available()
        if reason:
            raise EngineUnavailableError(f"Moteur basic_pitch non disponible. {reason}")
        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import Model

            self._model = Model(ICASSP_2022_MODEL_PATH)
        except Exception as exc:
            raise EngineUnavailableError(f"Moteur basic_pitch non disponible : {exc}") from exc
        suffix = Path(ICASSP_2022_MODEL_PATH).name
        self.backend = {
            "nmp": "tensorflow", "nmp.mlpackage": "coreml", "nmp.tflite": "tflite", "nmp.onnx": "onnx",
        }.get(suffix, suffix)  # fmt: skip
        self._loaded = True

    def transcribe(self, audio_path: Path) -> list[NoteEvent]:
        self.load()
        from basic_pitch.inference import predict

        try:
            _, _, note_events = predict(
                str(audio_path),
                self._model,
                onset_threshold=float(self.options.get("onset_threshold", 0.5)),
                frame_threshold=float(self.options.get("frame_threshold", 0.3)),
                minimum_note_length=float(self.options.get("minimum_note_length_ms", 58.0)),
            )
        except Exception as exc:
            raise TranscriptionError(f"Échec de la transcription Basic Pitch : {exc}") from exc
        events = [
            NoteEvent(
                onset=float(start),
                offset=float(end),
                pitch=int(pitch),
                velocity=int(round(127 * float(amplitude))),
                instrument=PIANO,
                confidence=float(amplitude),
            )
            for start, end, pitch, amplitude, *_ in note_events
        ]
        return normalize_events(events)
