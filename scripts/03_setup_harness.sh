#!/usr/bin/env bash
# Stage 0.2 — build the two benchmark harnesses in ISOLATED venvs.
#
# VLMEvalKit and lmms-eval pin conflicting dependencies, and both would clobber
# the torch build that stage 0 runs on. Each gets its own venv with its own
# torch; the base environment is never touched. The backends shell out, so the
# pins never meet.
#
# Expect this to take a while and several GB: each venv installs its own torch.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

read_cfg() {
  python - "$1" <<'PY'
import sys
from mmro.config import load_config
cfg = load_config()
h = cfg.eval.harnesses[sys.argv[1].split(".")[0]]
print(getattr(h, sys.argv[1].split(".")[1]))
PY
}

setup_one() {
  local name="$1"
  local venv repo repo_dir reqs
  venv="$(read_cfg "${name}.venv")"
  repo="$(read_cfg "${name}.repo")"
  repo_dir="$(read_cfg "${name}.repo_dir")"
  reqs="requirements/harness_${name}.txt"

  echo
  echo "############################################################"
  echo "# ${name}"
  echo "############################################################"

  if [[ -z "$repo" ]]; then
    echo "no repo configured for ${name}; skipping"
    return 0
  fi

  mkdir -p "$(dirname "$repo_dir")"
  if [[ -d "$repo_dir/.git" ]]; then
    echo "repo already cloned: $repo_dir"
  else
    echo "cloning $repo -> $repo_dir"
    git clone --depth 1 "$repo" "$repo_dir"
  fi
  # Pin what we actually ran. Harness CLIs and task definitions change often,
  # and an unpinned re-clone months later will not reproduce these scores.
  ( cd "$repo_dir" && git rev-parse HEAD > "${OLDPWD}/results/${name}_commit.txt" ) || true

  if [[ ! -x "$venv/bin/python" ]]; then
    echo "creating venv: $venv"
    python -m venv "$venv"
  fi
  "$venv/bin/python" -m pip install --quiet --upgrade pip

  echo "installing ${name} (this pulls its own torch; several GB)"
  "$venv/bin/python" -m pip install -e "$repo_dir"
  if [[ -f "$reqs" ]]; then
    "$venv/bin/python" -m pip install -r "$reqs"
  fi

  # The harnesses import mmro.models only if you later swap in our adapter.
  # Nothing here does that; the gate runs each harness's own implementation.
  echo "done: $("$venv/bin/python" --version)"
}

mkdir -p results
for h in vlmevalkit lmms_eval; do
  enabled="$(read_cfg "${h}.enabled")"
  if [[ "$enabled" != "True" ]]; then
    echo "skipping ${h} (disabled in config)"
    continue
  fi
  setup_one "$h"
done

echo
echo "############################################################"
echo "Next:"
echo "  1. Verify each harness's CLI against its README and, if it has changed,"
echo "     fix eval.harnesses.<name>.command in config.yaml -- not the code."
echo "  2. Set eval.judge.* -- published MathVista/MMMU numbers use LLM answer"
echo "     extraction, and rule-based fallback under-scores past the 2pt gate."
echo "  3. Shake out on a subset first:"
echo "       python -m mmro.eval --set eval.enabled=true --set eval.limit=25"
echo "  4. Then the real run: bash scripts/04_eval.sh"
