"""Moteur MT3 (Google Magenta, JAX/T5X) : transcription multi-instrument.

MT3 n'a pas de paquet PyPI ni d'API d'inférence officielle : le code ci-dessous reprend la classe
``InferenceModel`` du notebook officiel ``mt3/colab/music_transcription_with_transformers.ipynb``
(dépôt magenta/mt3, branche main). Seuls changent les chemins : les fichiers gin sont lus dans le
paquet ``mt3`` installé et le checkpoint dans ``models/mt3/checkpoints/<model_type>``.

L'installation est lourde et optionnelle (``requirements-mt3.txt``). Si un import échoue, le moteur
se déclare « non disponible » sans casser l'application.

TODO: vérifier — ce moteur n'a pas pu être exécuté sur la machine de développement (Windows :
``tensorflow-text``, dépendance de T5/seqio, n'y est pas distribué). Voir docs/models.md.
"""

from __future__ import annotations

import functools
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

SAMPLE_RATE = 16000
REQUIRED_MODULES = ("jax", "t5x", "seqio", "t5", "gin", "note_seq", "tensorflow", "mt3")
MODEL_TYPES = ("mt3", "ismir2021")
DEFAULT_VELOCITY = 80  # le modèle « mt3 » ne prédit pas les vélocités


def program_to_instrument(program: int, is_drum: bool) -> str:
    """Convertit un programme General MIDI en nom d'instrument normalisé."""
    if is_drum:
        return "drums"
    if 0 <= program <= 7:
        return PIANO
    try:
        import pretty_midi

        return str(pretty_midi.program_to_instrument_name(int(program))).lower()
    except Exception:
        return f"program_{program}"


class MT3Model(TranscriptionModel):
    """Transcription multi-instrument (ou piano seul avec ``model_type: ismir2021``)."""

    name = "mt3"
    supports_multi_instrument = True

    def __init__(self, options: dict[str, Any] | None = None, models_dir: Path | None = None) -> None:
        super().__init__(options, models_dir)
        self._inference: Any = None

    @property
    def model_type(self) -> str:
        """Variante du modèle : ``mt3`` ou ``ismir2021``."""
        return str(self.options.get("model_type") or "mt3")

    @property
    def checkpoint_dir(self) -> Path:
        """Dossier du checkpoint T5X."""
        custom = self.options.get("checkpoint_dir")
        if custom:
            return Path(custom).expanduser()
        return self.models_dir / "mt3" / "checkpoints" / self.model_type

    def check_available(self) -> str | None:
        if self.model_type not in MODEL_TYPES:
            return f"model_type inconnu : « {self.model_type} » (attendu : {', '.join(MODEL_TYPES)})."
        missing = [m for m in REQUIRED_MODULES if importlib.util.find_spec(m) is None]
        if missing:
            return (
                f"Dépendances absentes : {', '.join(missing)} "
                "(installation optionnelle : pip install -r requirements-mt3.txt, voir docs/models.md)."
            )
        if not (self.checkpoint_dir / "checkpoint").is_file():
            return f"Checkpoint absent : {self.checkpoint_dir} (python scripts/download_models.py --engine mt3)."
        return None

    def load(self) -> None:
        if self._loaded:
            return
        reason = self.check_available()
        if reason:
            raise EngineUnavailableError(f"Moteur mt3 non disponible. {reason}")
        try:
            self._inference = _build_inference_model(self.checkpoint_dir, self.model_type)
        except Exception as exc:
            raise EngineUnavailableError(f"Moteur mt3 non disponible : chargement impossible ({exc}).") from exc
        self._loaded = True

    def transcribe(self, audio_path: Path) -> list[NoteEvent]:
        self.load()
        import librosa

        try:
            audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
            sequence = self._inference(audio)
        except Exception as exc:
            raise TranscriptionError(f"Échec de la transcription MT3 : {exc}") from exc
        has_velocity = self.model_type == "ismir2021"
        events = [
            NoteEvent(
                onset=float(note.start_time),
                offset=float(note.end_time),
                pitch=int(note.pitch),
                velocity=int(note.velocity) if has_velocity and note.velocity else DEFAULT_VELOCITY,
                instrument=program_to_instrument(int(note.program), bool(note.is_drum)),
            )
            for note in sequence.notes
        ]
        return normalize_events(events)


def _build_inference_model(checkpoint_dir: Path, model_type: str) -> Any:
    """Construit l'``InferenceModel`` du notebook officiel (imports lourds, donc différés)."""
    import gin
    import jax
    import mt3
    import numpy as np
    import seqio
    import t5
    import t5x
    import tensorflow.compat.v2 as tf
    from mt3 import (
        metrics_utils,
        models,
        network,
        note_sequences,
        preprocessors,
        spectrograms,
        vocabularies,
    )

    gin_dir = Path(mt3.__file__).resolve().parent / "gin"

    class InferenceModel:
        """Enveloppe T5X pour l'inférence (reprise du notebook officiel)."""

        def __init__(self, checkpoint_path: str, model_type: str = "mt3") -> None:
            if model_type == "ismir2021":
                num_velocity_bins = 127
                self.encoding_spec = note_sequences.NoteEncodingSpec
                self.inputs_length = 512
            elif model_type == "mt3":
                num_velocity_bins = 1
                self.encoding_spec = note_sequences.NoteEncodingWithTiesSpec
                self.inputs_length = 256
            else:
                raise ValueError(f"unknown model_type: {model_type}")

            gin_files = [str(gin_dir / "model.gin"), str(gin_dir / f"{model_type}.gin")]
            self.batch_size = 8
            self.outputs_length = 1024
            self.sequence_length = {"inputs": self.inputs_length, "targets": self.outputs_length}
            self.partitioner = t5x.partitioning.PjitPartitioner(num_partitions=1)

            self.spectrogram_config = spectrograms.SpectrogramConfig()
            self.codec = vocabularies.build_codec(
                vocab_config=vocabularies.VocabularyConfig(num_velocity_bins=num_velocity_bins)
            )
            self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)
            self.output_features = {
                "inputs": seqio.ContinuousFeature(dtype=tf.float32, rank=2),
                "targets": seqio.Feature(vocabulary=self.vocabulary),
            }
            self._parse_gin(gin_files)
            self.model = self._load_model()
            self.restore_from_checkpoint(checkpoint_path)

        @property
        def input_shapes(self) -> dict[str, tuple[int, int]]:
            return {
                "encoder_input_tokens": (self.batch_size, self.inputs_length),
                "decoder_input_tokens": (self.batch_size, self.outputs_length),
            }

        def _parse_gin(self, gin_files: list[str]) -> None:
            gin_bindings = [
                "from __gin__ import dynamic_registration",
                "from mt3 import vocabularies",
                "VOCAB_CONFIG=@vocabularies.VocabularyConfig()",
                "vocabularies.VocabularyConfig.num_velocity_bins=%NUM_VELOCITY_BINS",
            ]
            with gin.unlock_config():
                gin.parse_config_files_and_bindings(gin_files, gin_bindings, finalize_config=False)

        def _load_model(self) -> Any:
            model_config = gin.get_configurable(network.T5Config)()
            module = network.Transformer(config=model_config)
            return models.ContinuousInputsEncoderDecoderModel(
                module=module,
                input_vocabulary=self.output_features["inputs"].vocabulary,
                output_vocabulary=self.output_features["targets"].vocabulary,
                optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),
                input_depth=spectrograms.input_depth(self.spectrogram_config),
            )

        def restore_from_checkpoint(self, checkpoint_path: str) -> None:
            train_state_initializer = t5x.utils.TrainStateInitializer(
                optimizer_def=self.model.optimizer_def,
                init_fn=self.model.get_initial_variables,
                input_shapes=self.input_shapes,
                partitioner=self.partitioner,
            )
            restore_checkpoint_cfg = t5x.utils.RestoreCheckpointConfig(
                path=checkpoint_path, mode="specific", dtype="float32"
            )
            train_state_axes = train_state_initializer.train_state_axes
            self._predict_fn = self._get_predict_fn(train_state_axes)
            self._train_state = train_state_initializer.from_checkpoint_or_scratch(
                [restore_checkpoint_cfg], init_rng=jax.random.PRNGKey(0)
            )

        @functools.lru_cache  # noqa: B019 - repris tel quel du notebook officiel
        def _get_predict_fn(self, train_state_axes: Any) -> Any:
            def partial_predict_fn(params: Any, batch: Any, decode_rng: Any) -> Any:
                return self.model.predict_batch_with_aux(params, batch, decoder_params={"decode_rng": None})

            return self.partitioner.partition(
                partial_predict_fn,
                in_axis_resources=(train_state_axes.params, t5x.partitioning.PartitionSpec("data"), None),
                out_axis_resources=t5x.partitioning.PartitionSpec("data"),
            )

        def predict_tokens(self, batch: Any, seed: int = 0) -> Any:
            prediction, _ = self._predict_fn(self._train_state.params, batch, jax.random.PRNGKey(seed))
            return self.vocabulary.decode_tf(prediction).numpy()

        def __call__(self, audio: Any) -> Any:
            ds = self.audio_to_dataset(audio)
            ds = self.preprocess(ds)
            model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(ds, task_feature_lengths=self.sequence_length)
            model_ds = model_ds.batch(self.batch_size)
            inferences = (
                tokens for batch in model_ds.as_numpy_iterator() for tokens in self.predict_tokens(batch)
            )
            predictions = [
                self.postprocess(tokens, example)
                for example, tokens in zip(ds.as_numpy_iterator(), inferences, strict=False)
            ]
            result = metrics_utils.event_predictions_to_ns(
                predictions, codec=self.codec, encoding_spec=self.encoding_spec
            )
            return result["est_ns"]

        def audio_to_dataset(self, audio: Any) -> Any:
            frames, frame_times = self._audio_to_frames(audio)
            return tf.data.Dataset.from_tensors({"inputs": frames, "input_times": frame_times})

        def _audio_to_frames(self, audio: Any) -> tuple[Any, Any]:
            frame_size = self.spectrogram_config.hop_width
            padding = [0, frame_size - len(audio) % frame_size]
            audio = np.pad(audio, padding, mode="constant")
            frames = spectrograms.split_audio(audio, self.spectrogram_config)
            num_frames = len(audio) // frame_size
            times = np.arange(num_frames) / self.spectrogram_config.frames_per_second
            return frames, times

        def preprocess(self, ds: Any) -> Any:
            pp_chain = [
                functools.partial(
                    t5.data.preprocessors.split_tokens_to_inputs_length,
                    sequence_length=self.sequence_length,
                    output_features=self.output_features,
                    feature_key="inputs",
                    additional_feature_keys=["input_times"],
                ),
                preprocessors.add_dummy_targets,
                functools.partial(preprocessors.compute_spectrograms, spectrogram_config=self.spectrogram_config),
            ]
            for pp in pp_chain:
                ds = pp(ds)
            return ds

        def postprocess(self, tokens: Any, example: Any) -> dict[str, Any]:
            tokens = self._trim_eos(tokens)
            start_time = example["input_times"][0]
            start_time -= start_time % (1 / self.codec.steps_per_second)
            return {"est_tokens": tokens, "start_time": start_time, "raw_inputs": []}

        @staticmethod
        def _trim_eos(tokens: Any) -> Any:
            tokens = np.array(tokens, np.int32)
            if vocabularies.DECODED_EOS_ID in tokens:
                tokens = tokens[: np.argmax(tokens == vocabularies.DECODED_EOS_ID)]
            return tokens

    return InferenceModel(str(checkpoint_dir) + "/", model_type)
