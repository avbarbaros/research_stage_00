#!/usr/bin/env bash
# Measure the OOM boundary for every enabled model.
#
# Each (model, image count, rung) cell runs in a fresh subprocess -- mmro.sweep
# handles that internally. Expect this to take a while: it deliberately runs
# cells that OOM, because that is the measurement.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

if ! python -m mmro.images >/dev/null 2>&1; then
  python -m mmro.images
  exit 1
fi

python -m mmro.sweep "$@"
status=$?

echo
python -m mmro.report
exit "$status"
