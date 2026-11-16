"""Interface commune des moteurs de transcription.

Tous les moteurs retournent des :class:`NoteEvent` normalisés : temps en secondes,
hauteur en numéro MIDI (60 = do central), vélocité 0-127, instrument renseigné.
Le pipeline aval ne dépend jamais du moteur.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PIANO = "piano"


class EngineUnavailableError(RuntimeError):
    """Le moteur ne peut pas être utilisé (dépendance ou poids manquants)."""


class TranscriptionError(RuntimeError):
    """La transcription a échoué (message destiné à l'utilisateur)."""


@dataclass
class NoteEvent:
    """Note transcrite, indépendante du moteur."""

    onset: float
    offset: float
    pitch: int
    velocity: int
    instrument: str = PIANO
    confidence: float | None = None

    @property
    def duration(self) -> float:
        """Durée en secondes."""
        return self.offset - self.onset

    def to_dict(self) -> dict[str, Any]:
        """Représentation sérialisable en JSON."""
        return {
            "onset": round(self.onset, 4),
            "offset": round(self.offset, 4),
            "pitch": self.pitch,
            "velocity": self.velocity,
            "instrument": self.instrument,
            "confidence": None if self.confidence is None else round(self.confidence, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NoteEvent:
        """Reconstruit une note depuis :meth:`to_dict`."""
        return cls(
            onset=float(data["onset"]),
            offset=float(data["offset"]),
            pitch=int(data["pitch"]),
            velocity=int(data["velocity"]),
            instrument=str(data.get("instrument", PIANO)),
            confidence=data.get("confidence"),
        )


@dataclass
class EngineStatus:
    """Disponibilité d'un moteur, affichée par l'API et la CLI."""

    name: str
    available: bool
    supports_multi_instrument: bool
    detail: str = ""


def normalize_events(events: list[NoteEvent]) -> list[NoteEvent]:
    """Borne les valeurs, écarte les notes dégénérées et trie par (début, hauteur)."""
    result: list[NoteEvent] = []
    for event in events:
        onset = max(0.0, float(event.onset))
        offset = float(event.offset)
        pitch = int(event.pitch)
        if offset <= onset or not 0 <= pitch <= 127:
            continue
        velocity = min(127, max(1, int(round(event.velocity))))
        result.append(
            NoteEvent(onset, offset, pitch, velocity, event.instrument or PIANO, event.confidence)
        )
    result.sort(key=lambda n: (n.onset, n.pitch))
    return result


class TranscriptionModel(ABC):
    """Classe de base d'un moteur de transcription."""

    name: str = ""
    supports_multi_instrument: bool = False

    def __init__(self, options: dict[str, Any] | None = None, models_dir: Path | None = None) -> None:
        self.options: dict[str, Any] = dict(options or {})
        self.models_dir: Path = Path(models_dir) if models_dir else Path("models")
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Charge les poids (appel paresseux, idempotent).

        Raises:
            EngineUnavailableError: dépendance ou poids manquants.
        """

    @abstractmethod
    def transcribe(self, audio_path: Path) -> list[NoteEvent]:
        """Transcrit un fichier audio en notes normalisées."""

    def check_available(self) -> str | None:
        """Retourne ``None`` si le moteur est utilisable, sinon la raison (sans charger les poids)."""
        return None

    def status(self) -> EngineStatus:
        """Disponibilité du moteur."""
        try:
            reason = self.check_available()
        except Exception as exc:  # une vérification ne doit jamais casser l'application
            reason = f"Vérification impossible : {exc}"
        return EngineStatus(
            name=self.name,
            available=reason is None,
            supports_multi_instrument=self.supports_multi_instrument,
            detail=reason or "Prêt",
        )
