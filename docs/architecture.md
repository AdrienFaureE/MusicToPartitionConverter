# Architecture

## Vue d'ensemble

```
            ┌────────────┐      ┌───────────────────┐
 navigateur │  web/      │ HTTP │  api/ (FastAPI)   │
 ─────────► │ index.html │ ───► │  routes, schemas  │
            └────────────┘      └─────────┬─────────┘
                                          │ submit / get
 terminal   ┌────────────┐      ┌─────────▼─────────┐
 ─────────► │ cli.py     │ ───► │ jobs.py           │  file de jobs (threads)
            └─────┬──────┘      └─────────┬─────────┘
                  │                       │
                  └───────────┬───────────┘
                              ▼
                      ┌───────────────┐
                      │ pipeline.py   │  orchestration d'un job
                      └───────┬───────┘
        ┌──────────┬──────────┼───────────┬──────────────┐
        ▼          ▼          ▼           ▼              ▼
     audio/   transcription/ postprocess/ notation/    render/
     load.py  base, registry cleanup      key_meter    pdf_export
              3 moteurs      tempo        musicxml_    html_render
                             quantize     builder
                             hand_split
```

Principe central : **les moteurs ne se parlent qu'à travers `NoteEvent`**. Tout ce qui suit la
transcription est identique quel que soit le moteur, ce qui permet de les comparer.

## Le pipeline, étape par étape

| # | Étape | Module | Entrée → sortie | Fichiers écrits |
| --- | --- | --- | --- | --- |
| 1 | Chargement audio | `audio/load.py` | fichier → mono, rééchantillonné, normalisé | `audio.wav` |
| 2 | Transcription | `transcription/*` | WAV → `list[NoteEvent]` | `raw.mid`, `raw_notes.json` |
| 3 | Nettoyage | `postprocess/cleanup.py` | notes → notes | `cleaned.mid` |
| 4 | Tempo et grille | `postprocess/tempo.py` | audio ou notes → `BeatGrid` | — |
| 5 | Quantification | `postprocess/quantize.py` | notes + grille → `list[QuantizedNote]` | `quantized.mid` |
| 6 | Tonalité et métrique | `notation/key_meter.py` | notes → tonalité, métrique | — |
| 7 | Séparation des mains | `postprocess/hand_split.py` | notes → `HandSplit` | `others.mid` s'il y a d'autres instruments |
| 8 | Construction | `notation/musicxml_builder.py` | `HandSplit` → partition music21 | — |
| 9 | Exports | `render/*` | partition → fichiers | `score.musicxml`, `score.html`, `score.pdf`, `report.json` |

Chaque étape est chronométrée (`report.json` → `timings_s`) et journalisée dans `job.log`.

### Reprise à partir d'une étape

`run_pipeline(..., start_from=...)` ou `partition-ia rebuild <dossier> --from ...` :

- `raw` : relit `raw.mid`, reprend au nettoyage (étapes 3 à 9) ;
- `quantized` : relit `quantized.mid` (tempo compris), reprend à la tonalité (étapes 6 à 9).

Usages : changer la grille, le seuil des mains, la tonalité ou la métrique sans retranscrire ;
corriger `quantized.mid` à la main dans un éditeur MIDI puis régénérer la partition.

## Choix de conception

### Nettoyage

- Vélocité inférieure au seuil → supprimée.
- Deux notes de même hauteur qui se chevauchent et commencent à moins de 50 ms d'écart → fusionnées
  (doublon de détection).
- Deux notes de même hauteur qui se chevauchent avec des attaques plus éloignées → la première est
  raccourcie (note rejouée). **Écart assumé par rapport à une fusion systématique** : fusionner
  ferait disparaître les notes répétées, fréquentes au piano. Un premier essai avec une fusion
  systématique supprimait 611 notes sur 1 563 dans la sortie de Basic Pitch.
- Durée inférieure à 30 ms → supprimée.

### Tempo et grille

`BeatGrid` contient l'instant de chaque noire. Deux modes :

- `tempo.method: audio` : suivi de battements `librosa`. Avec `follow_beats: true`, la grille suit
  les battements détectés : un rubato modéré n'empêche pas l'alignement. Le tempo affiché est la
  médiane, ramenée dans `[min_bpm, max_bpm]` par doublement ou division par deux.
- `tempo.method: midi` : autocorrélation des attaques des notes, tempo constant, première note
  calée sur la grille.

Limites connues : l'octave de tempo (60 ou 120) peut être mal choisie ; il n'y a pas de détection du
premier temps de la mesure ni de l'anacrouse : la pièce commence dans le premier temps de la mesure 1.
`tempo.forced_bpm` impose le tempo.

### Quantification

Positions en noires, en `Fraction` exactes. Grille par défaut : double croche ; option triolets de
croches (la position la plus proche des deux grilles gagne). Une note dure au moins un pas. Un
silence plus court que `fill_gap_beats` avant l'attaque suivante est absorbé (legato) ; les silences
plus longs sont conservés.

### Séparation des mains

- `threshold` : hauteur ≥ `split_pitch` (60, do central) → main droite.
- `chord` : par accord (notes attaquées ensemble). Accord jouable d'une main (étendue ≤
  `max_hand_span`) → il reste entier, du côté de sa hauteur moyenne. Accord plus large → coupé à son
  plus grand intervalle : voix hautes à droite, voix basses à gauche.
- Override par job : `hand_overrides.json` dans le dossier du job, ou `--hand-overrides`.

```json
{
  "split_pitch": 57,
  "method": "chord",
  "ranges": [
    {"start_measure": 5, "end_measure": 8, "split_pitch": 64},
    {"start_beat": 32, "end_beat": 36, "hand": "left"}
  ],
  "notes": [{"start_beat": 4.5, "pitch": 60, "hand": "left"}]
}
```

Priorité : note > intervalle (le dernier déclaré l'emporte) > réglage global. Les mesures sont
numérotées à partir de 1, bornes incluses ; les positions `start_beat` sont en noires depuis le début.

### Notation

Les mesures sont construites explicitement plutôt que par `makeNotation` de music21 :

- notes de même début et même durée → accord ;
- chevauchements → deuxième voix (jusqu'à `max_voices_per_staff`) ; au-delà, la note précédente est
  raccourcie ;
- notes à cheval sur une barre de mesure → découpées et liées ;
- silences découpés aux temps ; silences des voix secondaires masqués ;
- altérations : bémols dans les tonalités à bémols, dièses sinon.

Deux `PartStaff` regroupés par un `StaffGroup` donnent en MusicXML une seule partie « Piano » à deux
portées (`<staves>2</staves>`).

Métrique : 4/4 par défaut, forçable ; pas de détection automatique en V1.

### Rendu

- HTML : Verovio (paquet Python) produit un SVG par page, assemblé dans un `score.html` autonome,
  imprimable depuis le navigateur.
- PDF : MuseScore, sinon LilyPond, sinon rendu de secours interne (SVG Verovio convertis par
  `svglib`/`reportlab`). Le rendu de secours est vectoriel mais plus simple, et la figure de note du
  tempo peut manquer (police musicale non embarquée).

### Jobs et API

- `JobManager` : `ThreadPoolExecutor` à un seul travailleur (les modèles sont lourds en mémoire).
  État en mémoire, doublé d'un `job.json` par job pour survivre à un redémarrage en lecture seule.
- Un moteur : fichiers dans `data/outputs/<job_id>/`. Mode `all` : un sous-dossier par moteur ; un
  moteur indisponible n'interrompt pas les autres.
- L'API ne renvoie jamais de traceback : `PipelineError` porte un message lisible ; le détail
  technique reste dans `job.log` et dans la console du serveur.
- Le téléchargement de fichiers vérifie que le chemin demandé reste dans le dossier du job.

## Fichiers ajoutés par rapport à l'arborescence initiale

| Fichier | Raison |
| --- | --- |
| `src/partition/midi_io.py` | Lecture et écriture MIDI partagées par le pipeline, la reprise et les tests. |
| `LICENSE` | Présent dans le dépôt d'origine. |
| `.gitkeep` dans `models/`, `data/*`, `tests/fixtures/`, `web/vendor/` | Conserver les dossiers vides dans git. |

Le dépôt lui-même tient lieu de dossier `partition-ia/`.

## Compatibilité avec un déploiement futur

Aucun chemin absolu dans le code : tout est relatif à la racine du projet (`PARTITION_ROOT`) ou
configurable. Hôte, port et dossiers de données passent par des variables d'environnement. Pour un
hébergement type Render, il resterait à : ajouter un stockage persistant pour `data/` et `models/`,
remplacer la file en mémoire par une file externe, et limiter les moteurs à ceux que la mémoire de
l'instance supporte.
