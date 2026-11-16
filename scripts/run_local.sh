#!/usr/bin/env bash
# Lance l'API et l'interface web sur localhost.
# Utilisation : bash scripts/run_local.sh
# L'adresse et le port se règlent par PARTITION_SERVER__HOST et PARTITION_SERVER__PORT.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [ -f .venv/Scripts/activate ]; then
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
else
  echo "Environnement .venv absent : lancez d'abord scripts/setup_local.sh" >&2
  exit 1
fi

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
HOST="${PARTITION_SERVER__HOST:-127.0.0.1}"
PORT="${PARTITION_SERVER__PORT:-8000}"

echo "Interface : http://${HOST}:${PORT}/"
exec python -m uvicorn api.main:app --host "$HOST" --port "$PORT"
