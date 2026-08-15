#!/usr/bin/env bash
# Milestone criterion 5: config control is real.
#
# Greps src/ for values that must only ever live in config.yaml. If a model repo
# ID or an environment-specific path appears in source, then changing it in
# config.yaml would not change behaviour -- the exact failure this project is
# structured to avoid.
#
# Comments and docstrings legitimately name these things (that is documentation,
# not behaviour), so only string literals in code count. The greps below look
# for a quote character immediately before the value.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

status=0

check() {
  local label="$1" pattern="$2"
  local hits
  hits="$(grep -rnE "$pattern" src/ --include='*.py' | grep -vE '^\S+:[0-9]+:\s*#' || true)"
  if [[ -n "$hits" ]]; then
    echo "FAIL: $label"
    echo "$hits" | sed 's/^/    /'
    status=1
  else
    echo "ok  : $label"
  fi
}

check "no model repo IDs in src/"          "[\"'](Qwen/|OpenGVLab/)"
check "no absolute /content paths in src/" "[\"']/content"
check "no Drive paths in src/"             "[\"'][^\"']*MyDrive"
check "no hardcoded HF cache dirs in src/" "[\"'][^\"']*hf_cache"

echo
if [[ "$status" -eq 0 ]]; then
  echo "config control verified: repo IDs and paths live only in config.yaml"
else
  echo "config control BROKEN: move the values above into config.yaml"
fi
exit "$status"
