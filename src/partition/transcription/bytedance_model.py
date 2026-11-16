"""Moteur ByteDance (« High-resolution Piano Transcription », paquet ``piano_transcription_inference``).

API vérifiée dans le code de la version 0.0.6 :

- ``PianoTranscription(model_type='Note_pedal', checkpoint_path=None, segment_samples=160000, device=...)`` ;
- ``transcribe(audio, midi_path)`` attend un tableau mono à 16 kHz et retourne un dictionnaire
  dont ``est_note_events`` est une liste de ``{'onset_time', 'offset_time', 'midi_note', 'velocity'}``.

Le paquet télécharge lui-même le checkpoint avec ``wget`` (absent sous Windows) : nous passons
donc toujours un ``checkpoint_path`` explicite, téléchargé par ``scripts/download_models.py``.
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

CHECKPOINT_NAME = "CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"
CHECKPOINT_MIN_BYTES = 160_000_000  # le paquet lui-même exige plus de 1,6e8 octets
SAMPLE_RATE = 16000


class ByteDanceModel(TranscriptionModel):
    """Transcription piano seul, avec vélocités."""

    name = "bytedance"
    supports_multi_instrument = False

    def __init__(self, options: dict[str, Any] | None = None, models_dir: Path | None = None) -> None:
        super().__init__(options, models_dir)
        self._transcriptor: Any = None

    @property
    def checkpoint_path(self) -> Path:
        """Chemin du checkpoint (option ``checkpoint`` ou emplacement par défaut)."""
        custom = self.options.get("checkpoint")
        return Path(custom).expanduser() if custom else self.models_dir / "bytedance" / CHECKPOINT_NAME

    def check_available(self) -> str | None:
        for module in ("torch", "piano_transcription_inference"):
            if importlib.util.find_spec(module) is None:
                return f"Paquet Python « {module} » absent (pip install -r requirements.txt)."
        path = self.checkpoint_path
        if not path.is_file() or path.stat().st_size < CHECKPOINT_MIN_BYTES:
            return f"Checkpoint absent ou incomplet : {path} (python scripts/download_models.py --engine bytedance)."
        return None

    def load(self) -> None:
        if self._loaded:
            return
        reason = self.check_available()
        if reason:
            raise EngineUnavailableError(f"Moteur bytedance non disponible. {reason}")
        try:
            import torch
            from piano_transcription_inference import PianoTranscription
        except Exception as exc:
            raise EngineUnavailableError(f"Moteur bytedance non disponible : import impossible ({exc}).") from exc
        device = str(self.options.get("device") or "cpu")
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        try:
            self._transcriptor = PianoTranscription(device=device, checkpoint_path=str(self.checkpoint_path))
        except Exception as exc:
            raise EngineUnavailableError(f"Chargement du checkpoint ByteDance impossible : {exc}") from exc
        self._loaded = True

    def transcribe(self, audio_path: Path) -> list[NoteEvent]:
        self.load()
        import librosa

        try:
            audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
            result = self._transcriptor.transcribe(audio, None)
        except Exception as exc:
            raise TranscriptionError(f"Échec de la transcription ByteDance : {exc}") from exc
        events = [
            NoteEvent(
                onset=float(note["onset_time"]),
                offset=float(note["offset_time"]),
                pitch=int(note["midi_note"]),
                velocity=int(note["velocity"]),
                instrument=PIANO,
            )
            for note in result["est_note_events"]
        ]
        return normalize_events(events)
