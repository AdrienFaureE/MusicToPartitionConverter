"""Post-traitement des notes : nettoyage, tempo, quantification, séparation des mains."""

from partition.postprocess.cleanup import (
    apply_velocity_threshold,
    clean_notes,
    merge_overlapping,
    remove_short_notes,
)
from partition.postprocess.hand_split import (
    HandOverrideError,
    HandOverrides,
    HandSplit,
    load_overrides,
    parse_overrides,
    split_hands,
)
from partition.postprocess.quantize import QuantizedNote, grid_step, quantize_notes, snap
from partition.postprocess.tempo import (
    BeatGrid,
    constant_grid,
    estimate_grid,
    estimate_grid_from_audio,
    estimate_grid_from_notes,
    estimate_tempo_from_notes,
    fold_bpm,
)

__all__ = [
    "BeatGrid",
    "HandOverrideError",
    "HandOverrides",
    "HandSplit",
    "QuantizedNote",
    "apply_velocity_threshold",
    "clean_notes",
    "constant_grid",
    "estimate_grid",
    "estimate_grid_from_audio",
    "estimate_grid_from_notes",
    "estimate_tempo_from_notes",
    "fold_bpm",
    "grid_step",
    "load_overrides",
    "merge_overlapping",
    "parse_overrides",
    "quantize_notes",
    "remove_short_notes",
    "snap",
    "split_hands",
]
