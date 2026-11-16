"""Chargement de la configuration : YAML par défaut, puis surcharge par variables d'environnement.

Ordre de priorité (du plus faible au plus fort) :

1. ``config/default.yaml`` ;
2. le fichier désigné par ``PARTITION_CONFIG`` ou passé à :func:`load_settings` ;
3. les variables d'environnement ``PARTITION_*`` (``__`` sépare les niveaux,
   par exemple ``PARTITION_AUDIO__MAX_DURATION_S=300``) ;
4. les alias historiques ``MUSESCORE_PATH``, ``LILYPOND_PATH`` et ``FFMPEG_PATH``.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

ENV_PREFIX = "PARTITION_"
_RESERVED_ENV = {"PARTITION_CONFIG", "PARTITION_ROOT"}
_ALIASES = {
    "MUSESCORE_PATH": ("paths", "musescore_path"),
    "LILYPOND_PATH": ("paths", "lilypond_path"),
    "FFMPEG_PATH": ("paths", "ffmpeg_path"),
}


class ConfigError(ValueError):
    """Erreur de configuration (clé inconnue, valeur invalide, fichier absent)."""


@dataclass
class AudioSettings:
    """Paramètres de chargement audio."""

    sample_rate: int = 22050
    max_duration_s: float = 600.0
    normalize: bool = True
    peak: float = 0.95


@dataclass
class CleanupSettings:
    """Seuils de nettoyage des notes brutes."""

    min_duration_s: float = 0.03
    min_velocity: int = 12
    merge_window_s: float = 0.05


@dataclass
class TempoSettings:
    """Paramètres d'estimation du tempo."""

    method: str = "audio"
    forced_bpm: float | None = None
    min_bpm: float = 50.0
    max_bpm: float = 180.0
    follow_beats: bool = True


@dataclass
class QuantizeSettings:
    """Paramètres de la grille de quantification."""

    grid: int = 16
    triplets: bool = False
    fill_gap_beats: float = 0.25


@dataclass
class HandsSettings:
    """Paramètres de séparation des mains."""

    method: str = "threshold"
    split_pitch: int = 60
    max_hand_span: int = 14


@dataclass
class NotationSettings:
    """Paramètres de notation (tonalité, métrique, titre)."""

    key: str | None = None
    time_signature: str = "4/4"
    max_voices_per_staff: int = 2
    title: str | None = None
    composer: str | None = None
    author: str | None = None


@dataclass
class PathsSettings:
    """Chemins des dossiers et des exécutables externes."""

    models_dir: str = "models"
    data_dir: str = "data"
    musescore_path: str | None = None
    lilypond_path: str | None = None
    ffmpeg_path: str | None = None


@dataclass
class ServerSettings:
    """Paramètres du serveur local."""

    host: str = "127.0.0.1"
    port: int = 8000
    max_upload_mb: int = 100


@dataclass
class Settings:
    """Configuration complète de l'application."""

    engine: str = "bytedance"
    audio: AudioSettings = field(default_factory=AudioSettings)
    cleanup: CleanupSettings = field(default_factory=CleanupSettings)
    tempo: TempoSettings = field(default_factory=TempoSettings)
    quantize: QuantizeSettings = field(default_factory=QuantizeSettings)
    hands: HandsSettings = field(default_factory=HandsSettings)
    notation: NotationSettings = field(default_factory=NotationSettings)
    paths: PathsSettings = field(default_factory=PathsSettings)
    engines: dict[str, dict[str, Any]] = field(default_factory=dict)
    server: ServerSettings = field(default_factory=ServerSettings)
    root: Path = field(default_factory=lambda: project_root())

    # --- chemins résolus -------------------------------------------------
    def resolve(self, value: str | Path) -> Path:
        """Résout un chemin relatif par rapport à la racine du projet."""
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.root / path)

    @property
    def models_dir(self) -> Path:
        """Dossier des poids de modèles."""
        return self.resolve(self.paths.models_dir)

    @property
    def data_dir(self) -> Path:
        """Dossier des données (uploads et sorties)."""
        return self.resolve(self.paths.data_dir)

    @property
    def uploads_dir(self) -> Path:
        """Dossier des fichiers audio envoyés."""
        return self.data_dir / "uploads"

    @property
    def outputs_dir(self) -> Path:
        """Dossier des sorties, un sous-dossier par job."""
        return self.data_dir / "outputs"

    def engine_options(self, name: str) -> dict[str, Any]:
        """Options propres à un moteur (dictionnaire vide si absent)."""
        return dict(self.engines.get(name) or {})

    def to_dict(self) -> dict[str, Any]:
        """Représentation sérialisable (utile pour ``report.json``)."""
        data = dataclasses.asdict(self)
        data["root"] = str(self.root)
        return data


def project_root() -> Path:
    """Racine du projet : ``PARTITION_ROOT``, sinon le dossier contenant ``config/default.yaml``."""
    env_root = os.environ.get("PARTITION_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "config" / "default.yaml").is_file():
            return parent
    return Path.cwd()


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Fusionne récursivement deux dictionnaires sans modifier les originaux."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def env_overrides(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Construit le dictionnaire de surcharge à partir des variables d'environnement."""
    environ = os.environ if environ is None else environ
    result: dict[str, Any] = {}

    def assign(keys: tuple[str, ...], raw: str) -> None:
        node = result
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = _parse_scalar(raw)

    for name, keys in _ALIASES.items():
        if environ.get(name):
            assign(keys, environ[name])
    for name, raw in environ.items():
        if not name.startswith(ENV_PREFIX) or name in _RESERVED_ENV:
            continue
        nested = tuple(part.lower() for part in name[len(ENV_PREFIX) :].split("__") if part)
        if nested:
            assign(nested, raw)
    return result


def _parse_scalar(raw: str) -> Any:
    """Interprète une valeur d'environnement comme un scalaire YAML (nombre, booléen, null, texte)."""
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
    return raw if isinstance(value, (dict, list)) else value


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Fichier de configuration introuvable : {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML invalide dans {path} : {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} doit contenir un dictionnaire YAML.")
    return data


T = TypeVar("T")


def _build(cls: type[T], data: Mapping[str, Any], prefix: str = "") -> T:
    """Construit une dataclass depuis un dictionnaire, en refusant les clés inconnues."""
    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"Clé(s) de configuration inconnue(s) : {', '.join(prefix + k for k in unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        target = hints[name]
        if dataclasses.is_dataclass(target):
            if not isinstance(value, Mapping):
                raise ConfigError(f"La section « {prefix}{name} » doit être un dictionnaire.")
            kwargs[name] = _build(target, value, prefix=f"{prefix}{name}.")  # type: ignore[arg-type]
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _validate(settings: Settings) -> None:
    if settings.audio.sample_rate <= 0:
        raise ConfigError("audio.sample_rate doit être strictement positif.")
    if settings.audio.max_duration_s <= 0:
        raise ConfigError("audio.max_duration_s doit être strictement positif.")
    if settings.quantize.grid not in (4, 8, 16, 32):
        raise ConfigError("quantize.grid doit valoir 4, 8, 16 ou 32.")
    if settings.hands.method not in ("threshold", "chord"):
        raise ConfigError("hands.method doit valoir « threshold » ou « chord ».")
    if not 0 <= settings.hands.split_pitch <= 127:
        raise ConfigError("hands.split_pitch doit être une hauteur MIDI entre 0 et 127.")
    if settings.tempo.method not in ("audio", "midi"):
        raise ConfigError("tempo.method doit valoir « audio » ou « midi ».")
    if settings.tempo.min_bpm >= settings.tempo.max_bpm:
        raise ConfigError("tempo.min_bpm doit être inférieur à tempo.max_bpm.")
    if settings.notation.max_voices_per_staff < 1:
        raise ConfigError("notation.max_voices_per_staff doit être au moins 1.")


def load_settings(
    config_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Settings:
    """Charge la configuration complète.

    Args:
        config_path: fichier YAML supplémentaire, fusionné par-dessus ``config/default.yaml``.
        environ: variables d'environnement (``os.environ`` par défaut, injectable pour les tests).
        overrides: dictionnaire appliqué en dernier (options de la CLI, tests).
    """
    environ = os.environ if environ is None else environ
    root = project_root()
    data: dict[str, Any] = {}
    default_file = root / "config" / "default.yaml"
    if default_file.is_file():
        data = _read_yaml(default_file)
    extra = config_path or environ.get("PARTITION_CONFIG")
    if extra:
        data = deep_merge(data, _read_yaml(Path(extra).expanduser()))
    data = deep_merge(data, env_overrides(environ))
    if overrides:
        data = deep_merge(data, overrides)
    data.pop("root", None)
    settings = _build(Settings, data)
    settings.root = root
    _validate(settings)
    return settings
