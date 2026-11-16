# Moteurs de transcription

Trois moteurs partagent la même interface (`TranscriptionModel`, `NoteEvent`) et le même pipeline
aval. `partition-ia engines` ou `GET /api/engines` indique lesquels sont utilisables.

| | ByteDance | Basic Pitch | MT3 |
| --- | --- | --- | --- |
| Nom dans l'application | `bytedance` | `basic_pitch` | `mt3` |
| Cible | piano solo | toute source polyphonique | multi-instrument |
| Vélocités | oui | oui (amplitude) | non (`mt3`) / oui (`ismir2021`) |
| Distingue les instruments | non | non | oui |
| Bibliothèque | PyTorch | TensorFlow, ONNX, TFLite ou CoreML | JAX / T5X |
| Poids | 172 Mo, à télécharger | inclus dans le paquet pip | 164 Mo par variante, à télécharger |
| État sur le poste de développement | **exécuté** | **exécuté** | **non exécuté** (voir plus bas) |

## Temps de calcul mesurés

Mesures faites sur le poste de développement, **sur CPU**, avec un enregistrement de piano de
2 min 47 s (MP3 48 kHz) : Intel Core i7-1250U (12 cœurs logiques), 16 Go de RAM, Windows 11,
Python 3.11.8. Valeurs lues dans `report.json` (`transcription_s_per_audio_minute`, `engine_load_s`).

| Moteur | Calcul par minute d'audio | Chargement du modèle (une fois par processus) | Nombre de mesures |
| --- | --- | --- | --- |
| Basic Pitch (TensorFlow 2.15) | **1,8 s** (1,81 et 1,76) | 72 à 120 s au premier import de TensorFlow | 2 |
| ByteDance (PyTorch 2.15 CPU) | **129 s** (soit 2,1 fois la durée de l'audio) | 51 s, import de PyTorch compris | 1 |
| MT3 | non mesuré | non mesuré | 0 |

À lire avec prudence : une ou deux mesures, sur un processeur basse consommation, alors que d'autres
calculs tournaient parfois en parallèle. Les temps de chargement étaient ceux d'un premier lancement
(fichiers non encore en cache disque). Aucune mesure GPU n'a été faite. Chaque job réécrit ces
valeurs dans son `report.json` : c'est la référence pour votre machine.

Sur le même enregistrement : ByteDance a produit 1 309 notes brutes, Basic Pitch 1 563.

---

## ByteDance — `piano_transcription_inference`

- Dépôt : <https://github.com/qiuqiangkong/piano_transcription_inference> ; PyPI : `piano-transcription-inference` 0.0.6 (26 janvier 2025).
- Modèle : « High-resolution Piano Transcription with Pedals by Regressing Onset and Offset Times » (Kong et al.).

### Installation

Compris dans `requirements.txt` (`torch`, `piano_transcription_inference`). Le checkpoint :

```bash
python scripts/download_models.py --engine bytedance
```

- URL : `https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1`
- Destination : `models/bytedance/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth`
- Taille : 171 966 578 octets ; MD5 publié par Zenodo : `22b961b77c1878239fec963362097045` (vérifié par le script).

### API utilisée (lue dans le code de la version 0.0.6)

```python
from piano_transcription_inference import PianoTranscription, sample_rate  # sample_rate == 16000

transcriptor = PianoTranscription(device="cpu", checkpoint_path="models/bytedance/....pth")
result = transcriptor.transcribe(audio, None)   # audio : tableau mono à 16 kHz ; None = pas de fichier MIDI
result["est_note_events"]   # [{'onset_time', 'offset_time', 'midi_note', 'velocity'}, ...]
result["est_pedal_events"]  # pédale : ignorée en V1
```

Signature complète : `PianoTranscription(model_type='Note_pedal', checkpoint_path=None, segment_samples=160000, device=...)`.

### Différences avec la demande initiale

- La demande évoquait un téléchargement automatique par le paquet. C'est exact, mais il passe par
  `wget` (`os.system('wget ...')`), absent de Windows, et écrit dans le dossier personnel de
  l'utilisateur. Nous passons donc toujours un `checkpoint_path` explicite vers `models/`.
- Le paquet refuse un checkpoint de moins de 1,6 × 10⁸ octets ; notre vérification de disponibilité
  applique le même seuil.
- Le README officiel indique que Windows n'a pas été testé par les auteurs. Constat : fonctionne
  sous Windows 11 avec PyTorch 2.15 CPU.

### Réglages

`engines.bytedance.device` (`cpu` ou `cuda` ; retour automatique au CPU si CUDA est absent),
`engines.bytedance.checkpoint` (chemin personnalisé).

---

## Basic Pitch — `basic-pitch`

- Dépôt : <https://github.com/spotify/basic-pitch> ; PyPI : `basic-pitch` 0.4.0 (16 août 2024).

### Installation

Compris dans `requirements.txt`. Aucun poids à télécharger : le modèle ICASSP 2022 est livré dans le
paquet, dans quatre formats.

### Backend

Basic Pitch choisit le premier backend installé, dans cet ordre : TensorFlow, CoreML, TFLite, ONNX
(`basic_pitch.ICASSP_2022_MODEL_PATH`). Ce que `pip install basic-pitch` installe dépend de la
plateforme (métadonnées du paquet 0.4.0) :

| Plateforme | Python 3.10 | Python 3.11 |
| --- | --- | --- |
| Windows | `onnxruntime` → ONNX | `tensorflow<2.15.1` → TensorFlow |
| Linux | `tflite-runtime` → TFLite | `tensorflow<2.15.1` → TensorFlow |
| macOS | `coremltools` → CoreML | `coremltools` → CoreML |

**Backend utilisé sur le poste de développement : TensorFlow 2.15.0** (Windows, Python 3.11). Le
backend réellement utilisé est écrit dans `report.json` (`engine_backend`). Seul TensorFlow a été
exécuté ; ONNX, TFLite et CoreML suivent le même code mais n'ont pas été testés ici.

### API utilisée (lue dans le code de la version 0.4.0)

```python
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import Model, predict

model = Model(ICASSP_2022_MODEL_PATH)             # chargé une seule fois
model_output, midi_data, note_events = predict(
    audio_path, model,
    onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=58.0,  # millisecondes
)
# note_events : [(début_s, fin_s, hauteur_midi, amplitude 0-1, pitch_bends), ...]
```

L'amplitude est convertie en vélocité (`round(127 × amplitude)`) et conservée dans `confidence`.

### Points d'attention

- `resampy`, dépendance de Basic Pitch, importe `pkg_resources`, retiré de `setuptools` 81 :
  `requirements.txt` impose `setuptools<81`. Sans cela : `ModuleNotFoundError: No module named 'pkg_resources'`.
- TensorFlow 2.15 impose `numpy<2`.
- La valeur par défaut de `minimum_note_length` dans la bibliothèque est 127,7 ms ; nous utilisons
  58 ms (réglable par `engines.basic_pitch.minimum_note_length_ms`) pour garder les notes rapides.
- Basic Pitch n'est pas spécialisé piano : il détecte davantage de notes parasites que ByteDance
  (harmoniques, résonances).

---

## MT3 — Google Magenta

- Dépôt : <https://github.com/magenta/mt3> (version 0.0.1, « not an officially supported Google product »).
- Pas de paquet PyPI, pas d'API d'inférence publiée : la référence est le notebook
  `mt3/colab/music_transcription_with_transformers.ipynb`, dont nous reprenons la classe `InferenceModel`.

### État : non vérifié

MT3 **n'a pas pu être installé ni exécuté** sur le poste de développement (Windows) :
`tensorflow-text`, requis par `t5`/`seqio`, n'est pas distribué pour Windows. Le code de
`mt3_model.py` est une transcription fidèle du notebook officiel, mais **aucune ligne n'en a été
exécutée**. Ce qui a été vérifié : la détection d'indisponibilité (le moteur se déclare « non
disponible » avec la liste des dépendances manquantes, sans casser l'application) et la présence
des checkpoints sur le bucket public.

### Installation (Linux, macOS, ou WSL2 sous Windows)

```bash
pip install -r requirements-mt3.txt
python scripts/download_models.py --engine mt3
```

`requirements-mt3.txt` installe `jax`, `nest-asyncio` et `mt3` depuis GitHub. Le `setup.py` de MT3
tire à son tour, **sans épingler de version**, `flax`, `seqio`, `t5x` et `note-seq` depuis leurs
branches principales, ainsi que `t5`, `tensorflow`, `tensorflow-datasets`, `gin-config`, `librosa`.

Conséquences prévisibles :

- conflit probable avec TensorFlow 2.15 (épinglé par Basic Pitch) et `numpy<2` ;
- installation qui peut casser d'un jour à l'autre.

**Recommandation : un environnement virtuel dédié à MT3**, sans Basic Pitch. Les deux
environnements partagent `models/` et `data/` : on transcrit avec MT3 dans l'un, puis on compare
avec `partition-ia rebuild` ou l'interface dans l'autre. Le notebook officiel installe en plus
`libfluidsynth3 build-essential libasound2-dev libjack-dev` (paquets système) ; ils ne servent qu'à
la synthèse audio du notebook, pas à la transcription.

Versions testées : **aucune**. Si vous réussissez l'installation, notez la sortie de `pip freeze`.

### Checkpoints

Bucket public `gs://mt3/checkpoints`, lu sans authentification par l'API JSON de Google Cloud
Storage (`https://storage.googleapis.com/storage/v1/b/mt3/o?prefix=checkpoints/...`). Le notebook
utilise `gcloud storage cp` ; notre script évite cette dépendance et vérifie le MD5 de chaque objet.

| Variante | Contenu | Fichiers | Taille |
| --- | --- | --- | --- |
| `mt3` | multi-instrument, sans vélocités (ICLR 2022) | 295 | 164 Mo |
| `ismir2021` | piano seul, avec vélocités (ISMIR 2021) | 295 | 164 Mo |

Le bucket contient aussi `ismir2022_base` et `ismir2022_small`, non utilisés par le notebook.
Destination : `models/mt3/checkpoints/<variante>/`.

### Normalisation

- Programmes General MIDI 0 à 7 → `piano` ; autres → nom General MIDI en minuscules ; percussions → `drums`.
- La variante `mt3` ne prédit pas de vélocité : toutes les notes reçoivent 80.
- En V1, seules les notes `piano` sont notées ; les autres vont dans `others.mid`, avec un
  avertissement dans `report.json`.

### Différences avec la demande initiale

- La demande parlait de « checkpoints sur Google Cloud Storage » : confirmé (`gs://mt3/checkpoints`).
- Les fichiers gin sont lus dans le paquet installé (`<site-packages>/mt3/gin/`) et non dans
  `/content/mt3/gin/` comme dans le notebook.

---

## Ajouter un moteur

1. Créer une classe dérivée de `TranscriptionModel` (`name`, `supports_multi_instrument`, `load`,
   `transcribe`, et si possible `check_available`).
2. Retourner des `NoteEvent` passés par `normalize_events`.
3. L'enregistrer : `register_engine("nom", lambda options, models_dir: MonMoteur(options, models_dir))`.

Les imports lourds doivent rester dans `load()` pour que l'application démarre même si la
dépendance est absente.
