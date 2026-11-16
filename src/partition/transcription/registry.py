"""Registre des moteurs : sélection par nom, instances mises en cache."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from partition.config import Settings
from partition.transcription.base import EngineStatus, EngineUnavailableError, TranscriptionModel

EngineFactory = Callable[[dict[str, Any], Path], TranscriptionModel]

_factories: dict[str, EngineFactory] = {}
_instances: dict[tuple[str, str], TranscriptionModel] = {}
_lock = threading.Lock()


def register_engine(name: str, factory: EngineFactory) -> None:
    """Enregistre (ou remplace) un moteur."""
    with _lock:
        _factories[name] = factory
        for key in [k for k in _instances if k[0] == name]:
            del _instances[key]


def unregister_engine(name: str) -> None:
    """Retire un moteur (utilisé par les tests)."""
    with _lock:
        _factories.pop(name, None)
        for key in [k for k in _instances if k[0] == name]:
            del _instances[key]


def available_engine_names() -> list[str]:
    """Noms des moteurs enregistrés, dans l'ordre d'enregistrement."""
    return list(_factories)


def get_engine(name: str, settings: Settings) -> TranscriptionModel:
    """Retourne l'instance du moteur demandé (créée au premier appel, poids non chargés).

    Raises:
        EngineUnavailableError: nom inconnu.
    """
    if name not in _factories:
        known = ", ".join(available_engine_names()) or "aucun"
        raise EngineUnavailableError(f"Moteur inconnu : « {name} ». Moteurs disponibles : {known}.")
    key = (name, repr(sorted(settings.engine_options(name).items())) + str(settings.models_dir))
    with _lock:
        if key not in _instances:
            _instances[key] = _factories[name](settings.engine_options(name), settings.models_dir)
        return _instances[key]


def engine_statuses(settings: Settings) -> list[EngineStatus]:
    """Statut de chaque moteur enregistré."""
    return [get_engine(name, settings).status() for name in available_engine_names()]


def _register_builtin() -> None:
    """Enregistre les trois moteurs fournis. Les imports lourds restent paresseux."""
    from partition.transcription.basic_pitch_model import BasicPitchModel
    from partition.transcription.bytedance_model import ByteDanceModel
    from partition.transcription.mt3_model import MT3Model

    register_engine("bytedance", lambda opts, models: ByteDanceModel(opts, models))
    register_engine("mt3", lambda opts, models: MT3Model(opts, models))
    register_engine("basic_pitch", lambda opts, models: BasicPitchModel(opts, models))


_register_builtin()
