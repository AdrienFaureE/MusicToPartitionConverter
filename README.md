# partition-ia

Transcription locale d'un enregistrement de piano en partition : **MusicXML** (éditable dans
MuseScore), **PDF** et aperçu **HTML**, avec séparation main droite / main gauche. Tout tourne sur
votre machine, sans service externe.

- Entrée : MP3, WAV, FLAC ou M4A de 3 à 5 minutes (10 minutes au maximum par défaut), piano solo de préférence.
- Trois moteurs de transcription comparables sur la même chaîne aval : `bytedance`, `basic_pitch`, `mt3`.
- Une interface web locale, une API et une ligne de commande.

> V1 : seul le piano est noté. Les autres instruments détectés (moteur MT3) sont enregistrés dans un
> MIDI séparé.

## Pipeline

```
audio ──► 1. chargement ──► 2. transcription ──► 3. nettoyage ──► 4. tempo et grille
          (mono, WAV)       (bytedance │          (notes courtes,   (librosa ou notes)
                             basic_pitch │ mt3)    doublons, vélocité)
                                  │ raw.mid
                                  ▼
5. quantification ──► 6. tonalité, métrique ──► 7. mains ──► 8. music21 ──► 9. exports
   (1/16, triolets)      (Krumhansl, 4/4)         (seuil, accords,  (2 portées,   score.musicxml
       │ quantized.mid                             override JSON)    voix, liaisons) score.pdf
                                                                                    score.html
                                                                                    report.json
```

Chaque étape écrit ses fichiers dans `data/outputs/<job_id>/` et son journal dans `job.log`.
Le pipeline peut repartir de `raw.mid` ou de `quantized.mid` sans retranscrire.
Détails : [docs/architecture.md](docs/architecture.md).

## Installation

Prérequis : **Python 3.10 ou 3.11**. MuseScore 4 est recommandé pour le PDF, ffmpeg pour le M4A.
Liste complète : [docs/prerequisites.md](docs/prerequisites.md).

### Windows

```powershell
git clone https://github.com/AdrienFaureE/MusicToPartitionConverter.git
cd MusicToPartitionConverter
powershell -ExecutionPolicy Bypass -File scripts\setup_local.ps1
```

### macOS et Linux

```bash
git clone https://github.com/AdrienFaureE/MusicToPartitionConverter.git
cd MusicToPartitionConverter
bash scripts/setup_local.sh            # ajouter --with-mt3 pour tenter l'installation de MT3
```

Sous Linux sans GPU, installez d'abord PyTorch en version CPU pour éviter plusieurs Go de
bibliothèques CUDA : `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
Paquets système utiles : `sudo apt install libsndfile1 ffmpeg` (Debian/Ubuntu), `brew install ffmpeg` (macOS).

### Installation manuelle (tous systèmes)

```bash
python -m venv .venv
source .venv/bin/activate                    # Windows : .venv\Scripts\activate
python -m pip install --upgrade pip "setuptools<81"
pip install -r requirements.txt -r requirements-dev.txt
pip install --no-deps -e .
python scripts/download_models.py            # checkpoint ByteDance (172 Mo)
python scripts/check_prerequisites.py
```

Les scripts font exactement cela : création du venv, installation, téléchargement des poids,
vérification. MT3 est optionnel et s'installe à part : [docs/models.md](docs/models.md).

## Lancement

```bash
bash scripts/run_local.sh                    # macOS, Linux, Git Bash
```

```powershell
.venv\Scripts\python.exe -m partition.cli serve      # Windows
```

Puis ouvrir <http://127.0.0.1:8000/>. Documentation interactive de l'API : <http://127.0.0.1:8000/docs>.

## Interface web

1. Choisir un fichier audio.
2. Choisir un moteur. Les moteurs indisponibles sont grisés, avec la raison. « Tous les moteurs »
   les lance l'un après l'autre pour comparaison.
3. Cliquer sur « Transcrire ». La barre de progression suit les étapes.
4. À la fin : téléchargements (PDF, MusicXML, MIDI brut et quantifié, rapport, journal) et aperçu de
   la partition dans la page. Avec plusieurs moteurs, les partitions s'affichent côte à côte.

L'adresse `/?job=<identifiant>` rouvre un job déjà traité.

## Ligne de commande

```bash
partition-ia transcribe morceau.mp3 --engine basic_pitch --output data/outputs/test
partition-ia transcribe morceau.mp3 --engine all                  # un sous-dossier par moteur
partition-ia rebuild data/outputs/test --from raw                 # reprend au nettoyage
partition-ia rebuild data/outputs/test --from quantized           # reprend à la tonalité
partition-ia transcribe morceau.mp3 --hand-overrides mains.json   # séparation des mains forcée
partition-ia engines                                              # moteurs et disponibilité
partition-ia serve                                                # API et interface
```

Sans installation du paquet : `PYTHONPATH=src python -m partition.cli ...`.

Réglages ponctuels par variables d'environnement, par exemple pour reconstruire en 3/4 avec un seuil
de mains plus bas :

```bash
PARTITION_NOTATION__TIME_SIGNATURE=3/4 PARTITION_HANDS__SPLIT_PITCH=55 partition-ia rebuild data/outputs/test
```

## Configuration

`config/default.yaml` contient toutes les options, commentées. Les variables d'environnement
`PARTITION_<SECTION>__<CLÉ>` les surchargent ; elles sont listées dans [.env.example](.env.example).
`MUSESCORE_PATH`, `LILYPOND_PATH` et `FFMPEG_PATH` sont acceptés tels quels.

## API

| Méthode | Route | Rôle |
| --- | --- | --- |
| `POST` | `/api/jobs` | Envoie un fichier (`file`) et un moteur (`engine` : `bytedance`, `mt3`, `basic_pitch`, `all`). Renvoie `job_id`. |
| `GET` | `/api/jobs/{id}` | État, progression, résultats par moteur. |
| `GET` | `/api/jobs/{id}/files/{name}` | Télécharge un fichier (`score.pdf`, `score.musicxml`, `raw.mid`, `quantized.mid`, `score.html`, `report.json`, `job.log`). En mode `all` : `{moteur}/{name}`. |
| `GET` | `/api/engines` | Moteurs et disponibilité. |
| `GET` | `/api/health` | État du serveur. |

## Comparaison des moteurs

| | ByteDance | Basic Pitch | MT3 |
| --- | --- | --- | --- |
| Cible | piano solo | toute source polyphonique | multi-instrument |
| Qualité attendue sur piano | la meilleure des trois | correcte, plus de notes parasites | bonne, sans vélocités |
| Calcul par minute d'audio (CPU, mesuré) | 129 s | 1,8 s | non mesuré |
| Installation | pip + checkpoint de 172 Mo | pip seul | lourde, Linux/macOS, à part |
| Vérifié sur le poste de développement | oui | oui | non |

Mesures, conditions et détails d'installation : [docs/models.md](docs/models.md).

Pour comparer sur un même morceau : moteur « Tous les moteurs » dans l'interface, ou
`partition-ia transcribe morceau.mp3 --engine all`. Chaque `report.json` donne le tempo, la
tonalité, le nombre de notes à chaque étape, les temps de calcul et les avertissements.

## Tests

```bash
pytest tests/unit tests/integration/test_api.py   # rapides, sans poids de modèles (environ 20 s)
pytest -m integration                             # moteurs réels ; ignorés si poids ou dépendances absents
ruff check .
mypy
```

Les fixtures (MIDI et audio synthétiques de 10 secondes) sont générées par
`tests/fixtures_generator.py` au premier lancement ; aucun binaire n'est versionné.

## Dépannage

| Symptôme | Cause et solution |
| --- | --- |
| « Impossible de décoder … Installez ffmpeg » | Format non lu par libsndfile et ffmpeg introuvable. Installer ffmpeg ou `pip install imageio-ffmpeg`, ou définir `FFMPEG_PATH`. |
| Avertissement « PDF produit par le rendu de secours Verovio » | MuseScore introuvable. L'installer, ou définir `MUSESCORE_PATH` vers l'exécutable (`MuseScore4.exe`, `mscore`). Le MusicXML reste ouvrable dans MuseScore. |
| `ModuleNotFoundError: No module named 'pkg_resources'` | `setuptools` trop récent : `pip install "setuptools<81"`. |
| Conflit TensorFlow / JAX / NumPy après l'installation de MT3 | Installer MT3 dans un environnement virtuel séparé (voir docs/models.md) et réinstaller `requirements.txt` dans l'environnement principal. |
| Moteur « indisponible » dans l'interface | La raison est affichée sous son nom et par `partition-ia engines` : paquet absent, ou checkpoint à télécharger avec `python scripts/download_models.py`. |
| Mémoire insuffisante, processus tué | Utiliser `basic_pitch`, éviter le mode « Tous les moteurs », fermer les autres applications, réduire `PARTITION_AUDIO__MAX_DURATION_S`. |
| Premier job très lent à démarrer | Premier import de TensorFlow ou de PyTorch (jusqu'à deux minutes mesurées). Les jobs suivants du même serveur réutilisent le modèle chargé. |
| Tempo doublé ou divisé par deux | Ambiguïté d'octave du suivi de battements : imposer `PARTITION_TEMPO__FORCED_BPM`, puis `partition-ia rebuild <dossier> --from raw`. |
| Mains mal réparties | Changer `PARTITION_HANDS__SPLIT_PITCH`, essayer `PARTITION_HANDS__METHOD=chord`, ou écrire un `hand_overrides.json` (format dans docs/architecture.md). |
| Partition trop chargée, rythmes illisibles | Essayer `PARTITION_QUANTIZE__GRID=8`, relever `PARTITION_CLEANUP__MIN_VELOCITY`, ou corriger `quantized.mid` puis `rebuild --from quantized`. |

## Limites connues de la V1

- La transcription automatique n'est pas une partition d'éditeur : attendez-vous à retoucher le
  MusicXML dans MuseScore (rythmes, voix, répartition des mains).
- Pas de détection de la métrique, de l'anacrouse ni du premier temps : 4/4 par défaut, forçable.
- La pédale, les nuances et les articulations ne sont pas notées.
- MT3 n'a pas été exécuté pendant le développement.

## Feuille de route

- **V2** : notation multi-instruments (une portée par instrument détecté par MT3), détection de la
  métrique et de l'anacrouse, pédale, séparation des mains par suivi de voix.
- **Déploiement Render** : stockage persistant pour `data/` et `models/`, file de jobs externe,
  choix des moteurs selon la mémoire de l'instance. La configuration passe déjà par des variables
  d'environnement et aucun chemin n'est codé en dur.

## Structure du dépôt

```
config/default.yaml        configuration par défaut
src/partition/             pipeline (audio, transcription, postprocess, notation, render, jobs, cli)
api/                       application FastAPI
web/                       interface (HTML, CSS, JS sans framework)
scripts/                   installation, téléchargement des poids, vérification, lancement
tests/                     tests unitaires, d'intégration et générateur de fixtures
docs/                      architecture, moteurs, prérequis
models/, data/             poids et données des jobs (ignorés par git)
```

Fichiers ajoutés par rapport à l'arborescence prévue, et pourquoi : voir la fin de
[docs/architecture.md](docs/architecture.md).

## Licence

MIT, voir [LICENSE](LICENSE). Les modèles et leurs poids ont leurs propres licences : consultez leurs
dépôts (liens dans [docs/models.md](docs/models.md)) avant toute diffusion.
