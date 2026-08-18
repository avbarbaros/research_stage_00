#!/usr/bin/env bash
# Stage 0.2 — run both harnesses on every configured task, then check the gate.
#
# This is long: MMMU val (~900 items) + MathVista testmini (1000) across two
# harnesses is roughly 3,800 generations. Budget hours, not minutes, and run it
# where a dropped Colab session will not cost you the whole thing -- scores are
# appended per task, so a crash keeps everything already finished.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

python -m mmro.eval "$@"
status=$?

echo
python -m mmro.report
exit "$status"
