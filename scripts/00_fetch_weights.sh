#!/usr/bin/env bash
# Download and stage weights for every enabled model.
#
# Resumable: an interrupted 16GB pull continues rather than restarting, so
# re-running after a dropped Colab session is the correct response.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

python -m mmro.cache "$@"
