# Prérequis

`python scripts/check_prerequisites.py` vérifie tout ce qui suit et indique ce qui manque.

## Indispensable

| Élément | Version | Remarques |
| --- | --- | --- |
| Python | **3.10 ou 3.11** | Testé avec 3.11.8 (Windows 11). Voir « Python 3.12 » ci-dessous. |
| Paquets Python | `requirements.txt` | Installés par `scripts/setup_local.sh` ou `.ps1`. |
| Au moins un moteur | — | Basic Pitch est prêt dès l'installation ; ByteDance demande son checkpoint (`scripts/download_models.py`). |

### Python 3.12

Non pris en charge. Constat fait à l'installation : `basic-pitch` 0.4.0 épingle TensorFlow 2.15
(`tensorflow<2.15.1,>=2.4.1` dès Python 3.11), et TensorFlow 2.15 n'est publié que pour Python 3.9 à 3.11. Le fichier
`pyproject.toml` impose donc `>=3.10,<3.12`. Non testé de notre côté sous 3.12 : l'installation
des moteurs y échouerait à la résolution des dépendances.

## Recommandé

| Élément | Rôle | Sans lui |
| --- | --- | --- |
| **MuseScore 4** (CLI) | PDF de référence (`mscore -o score.pdf score.musicxml`) | Le PDF est produit par le rendu de secours interne (Verovio), plus simple ; un avertissement est écrit dans `report.json`. |
| **ffmpeg** dans le PATH | Décodage des formats non lus par libsndfile (M4A) | Le binaire fourni par le paquet `imageio-ffmpeg` est utilisé automatiquement. MP3, WAV et FLAC sont lus sans ffmpeg. |
| 16 Go de RAM | Chargement des modèles | 8 Go suffisent pour Basic Pitch et ByteDance sur des morceaux de 3 à 5 minutes ; MT3 est plus gourmand (non mesuré). |

## Optionnel

| Élément | Rôle |
| --- | --- |
| LilyPond | Secours pour le PDF si MuseScore est absent (`musicxml2ly` puis `lilypond`). |
| GPU CUDA | Accélère ByteDance (`PARTITION_ENGINES__BYTEDANCE__DEVICE=cuda`) et MT3. Tout fonctionne sur CPU. |
| Git | Clonage du dépôt et installation de MT3 (depuis GitHub). |

## Installation des outils externes

### ffmpeg

- **Windows** : `winget install Gyan.FFmpeg`, puis rouvrir le terminal.
- **macOS** : `brew install ffmpeg`.
- **Linux (Debian/Ubuntu)** : `sudo apt install ffmpeg libsndfile1`.

Chemin explicite possible : `FFMPEG_PATH` ou `paths.ffmpeg_path`.

### MuseScore 4

Télécharger MuseScore Studio 4 sur <https://musescore.org/fr/download>.

Détection automatique, dans cet ordre :

1. `MUSESCORE_PATH` ou `paths.musescore_path` ;
2. le PATH : `mscore`, `mscore4`, `musescore`, `musescore4`, `MuseScore4`, `mscore4portable`, `mscore3`, `MuseScore3` ;
3. les emplacements habituels :
   - Windows : `%ProgramFiles%\MuseScore 4\bin\MuseScore4.exe` (et variantes « MuseScore Studio 4 », `%LOCALAPPDATA%\Programs`) ;
   - macOS : `/Applications/MuseScore 4.app/Contents/MacOS/mscore` (et « MuseScore Studio.app ») ;
   - Linux : `/usr/bin/mscore`, `/usr/local/bin/mscore`, `/snap/bin/musescore`, un `*.AppImage` dans le dossier personnel.

Sous Linux sans écran, `QT_QPA_PLATFORM=offscreen` est défini automatiquement pour l'appel.

> À valider sur votre machine : MuseScore n'était pas installé sur le poste de développement.
> La commande et la détection suivent la documentation de MuseScore mais n'ont pas été exécutées.

### LilyPond

<https://lilypond.org/download.html>, ou `brew install lilypond`, `sudo apt install lilypond`.
Chemin explicite : `LILYPOND_PATH`. Non exécuté sur le poste de développement.

## Espace disque

| Élément | Taille |
| --- | --- |
| Environnement Python (TensorFlow, PyTorch CPU, etc.) | 2,6 Go (mesuré sous Windows, avec les outils de développement) |
| Checkpoint ByteDance | 172 Mo |
| Checkpoints MT3 (`mt3` et `ismir2021`) | 164 Mo chacun (295 fichiers) |
| Sorties d'un job (WAV de travail compris) | 9,6 Mo mesurés pour 2 min 47 s d'audio |
