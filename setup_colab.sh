#!/usr/bin/env bash
# Bootstrap the stage-0 environment on Colab.
#
# Deliberately does NOT install torch. Colab ships a torch build matched to its
# CUDA driver; reinstalling it is the single most common way to break this
# setup. We install only the layer above and record what we resolved to.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

echo "=== 1/4  config ==="
if [[ ! -f config.yaml ]]; then
  cp config.example.yaml config.yaml
  echo "created config.yaml from config.example.yaml -- edit it before running anything else"
else
  echo "config.yaml already exists; leaving it alone"
fi

echo
echo "=== 2/4  dependencies ==="
python -m pip install --quiet --upgrade pip
# --upgrade-strategy only-if-needed keeps pip from pulling a new torch in as a
# transitive upgrade.
python -m pip install --quiet --upgrade-strategy only-if-needed -r requirements/base.txt
echo "installed requirements/base.txt (torch left untouched)"

echo
echo "=== 3/4  config validation ==="
python -m mmro.config --validate

echo
echo "=== 4/4  environment probe ==="
python -m mmro.env "$@"

echo
echo "Next:"
echo "  1. copy your held-out images into the directory shown by:  python -m mmro.images"
echo "  2. bash scripts/00_fetch_weights.sh"
echo "  3. bash scripts/01_smoke.sh"
