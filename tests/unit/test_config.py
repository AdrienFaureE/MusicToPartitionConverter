"""Tests de la configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from partition.config import (
    ConfigError,
    deep_merge,
    env_overrides,
    load_settings,
    project_root,
)


def test_defaults_from_yaml() -> None:
    settings = load_settings(environ={})
    assert settings.engine == "bytedance"
    assert settings.audio.max_duration_s == 600
    assert settings.cleanup.min_duration_s == pytest.approx(0.03)
    assert settings.quantize.grid == 16
    assert settings.hands.split_pitch == 60
    assert settings.notation.time_signature == "4/4"
    assert settings.notation.key is None


def test_env_overrides_yaml() -> None:
    environ = {
        "PARTITION_ENGINE": "basic_pitch",
        "PARTITION_AUDIO__MAX_DURATION_S": "300",
        "PARTITION_QUANTIZE__TRIPLETS": "true",
        "PARTITION_NOTATION__KEY": "a minor",
        "PARTITION_ENGINES__BYTEDANCE__DEVICE": "cuda",
        "MUSESCORE_PATH": "/opt/mscore",
        "AUTRE_VARIABLE": "ignorée",
    }
    settings = load_settings(environ=environ)
    assert settings.engine == "basic_pitch"
    assert settings.audio.max_duration_s == 300
    assert settings.quantize.triplets is True
    assert settings.notation.key == "a minor"
    assert settings.engine_options("bytedance")["device"] == "cuda"
    assert settings.paths.musescore_path == "/opt/mscore"


def test_prefixed_variable_wins_over_alias() -> None:
    result = env_overrides({"MUSESCORE_PATH": "a", "PARTITION_PATHS__MUSESCORE_PATH": "b"})
    assert result["paths"]["musescore_path"] == "b"


def test_extra_yaml_file(tmp_path: Path) -> None:
    extra = tmp_path / "local.yaml"
    extra.write_text("hands:\n  split_pitch: 55\n", encoding="utf-8")
    settings = load_settings(config_path=extra, environ={})
    assert settings.hands.split_pitch == 55
    assert settings.hands.method == "threshold"


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="inconnue"):
        load_settings(environ={"PARTITION_AUDIO__INCONNU": "1"})


@pytest.mark.parametrize(
    "overrides",
    [
        {"quantize": {"grid": 7}},
        {"hands": {"method": "magie"}},
        {"hands": {"split_pitch": 200}},
        {"audio": {"max_duration_s": 0}},
        {"tempo": {"min_bpm": 200, "max_bpm": 100}},
    ],
)
def test_invalid_values_are_rejected(overrides: dict) -> None:
    with pytest.raises(ConfigError):
        load_settings(environ={}, overrides=overrides)


def test_missing_config_file() -> None:
    with pytest.raises(ConfigError, match="introuvable"):
        load_settings(config_path="n_existe_pas.yaml", environ={})


def test_paths_are_resolved_from_root() -> None:
    settings = load_settings(environ={})
    assert settings.root == project_root()
    assert settings.models_dir == settings.root / "models"
    assert settings.outputs_dir == settings.root / "data" / "outputs"
    assert settings.uploads_dir == settings.root / "data" / "uploads"
    assert settings.resolve("/abs" if Path("/abs").is_absolute() else "C:/abs").is_absolute()
    assert isinstance(settings.to_dict()["root"], str)


def test_deep_merge_does_not_mutate() -> None:
    base = {"a": {"b": 1, "c": 2}}
    merged = deep_merge(base, {"a": {"b": 3}})
    assert merged == {"a": {"b": 3, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}
