#!/usr/bin/env bash
# Installation locale (macOS / Linux) : venv, dépendances, vérifications, poids des modèles.
# Utilisation : bash scripts/setup_local.sh [--with-mt3] [--skip-models]
set -euo pipefail

cd "$(dirname "$0")/.."

WITH_MT3=""
SKIP_MODELS=""
for arg in "$@"; do
  case "$arg" in
    --with-mt3) WITH_MT3=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    *) echo "Option inconnue : $arg" >&2; exit 2 ;;
  esac
done

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for candidate in python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$candidate"; break; fi
  done
fi
if [ -z "$PYTHON" ]; then
  echo "Python introuvable. Installez Python 3.10 ou 3.11 (voir docs/prerequisites.md)." >&2
  exit 1
fi
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2] in ((3, 10), (3, 11)) else 1)'; then
  echo "Python 3.10 ou 3.11 requis (trouvé : $("$PYTHON" --version)). Indiquez-le avec PYTHON=/chemin/python3.11." >&2
  exit 1
fi

echo "== Environnement virtuel (.venv) =="
[ -d .venv ] || "$PYTHON" -m venv .venv
# shellcheck disable=SC1091
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; else source .venv/Scripts/activate; fi
python -m pip install --upgrade pip "setuptools<81" wheel

echo "== Dépendances communes =="
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .

if [ -z "$WITH_MT3" ] && [ -t 0 ]; then
  read -r -p "Installer MT3 (JAX/T5X, lourd, Linux/macOS uniquement) ? [o/N] " answer
  case "$answer" in o|O|oui|y|Y) WITH_MT3=1 ;; esac
fi
if [ -n "$WITH_MT3" ]; then
  echo "== MT3 (optionnel) =="
  python -m pip install -r requirements-mt3.txt || echo "Installation de MT3 en échec : le moteur restera indisponible (voir docs/models.md)."
fi

if [ -z "$SKIP_MODELS" ]; then
  echo "== Poids des modèles =="
  if [ -n "$WITH_MT3" ]; then python scripts/download_models.py --engine all; else python scripts/download_models.py; fi
fi

echo "== Vérification des prérequis =="
python scripts/check_prerequisites.py || true

echo
echo "Installation terminée. Lancez l'application avec : bash scripts/run_local.sh"
