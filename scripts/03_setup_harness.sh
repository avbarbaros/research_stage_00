#!/usr/bin/env bash
# Stage 0.2 — build the two benchmark harnesses in ISOLATED venvs.
#
# VLMEvalKit and lmms-eval pin conflicting dependencies, and both would clobber
# the torch build that stage 0 runs on. Each gets its own venv with its own
# torch; the base environment is never touched. The backends shell out, so the
# pins never meet.
#
# THE VENVS MUST BE ON LOCAL DISK. A virtualenv needs symlinks and `ensurepip`
# needs normal file semantics; Google Drive's FUSE mount provides neither, and
# `python -m venv` dies with an ensurepip error. A git clone succeeds there
# because it only writes regular files, which is why the failure shows up
# halfway through. This script checks before installing anything.
#
# Local disk is wiped between Colab sessions, so each venv (~5GB, mostly torch)
# would be rebuilt every time. Pass --archive to tar them to Drive afterwards;
# subsequent runs restore in a couple of minutes instead of reinstalling.
#
#   bash scripts/03_setup_harness.sh            # build (restores from archive if present)
#   bash scripts/03_setup_harness.sh --archive  # build, then save to Drive
#   bash scripts/03_setup_harness.sh --rebuild  # ignore any archive, install fresh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$PWD"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

ARCHIVE_DIR="${MMRO_VENV_ARCHIVE_DIR:-${REPO_ROOT}/.venv_archives}"
DO_ARCHIVE=0
REBUILD=0
for arg in "$@"; do
  case "$arg" in
    --archive) DO_ARCHIVE=1 ;;
    --rebuild) REBUILD=1 ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

read_cfg() {
  python - "$1" <<'PY'
import sys
from mmro.config import load_config
name, field = sys.argv[1].split(".", 1)
print(getattr(load_config().eval.harnesses[name], field))
PY
}

# --- the check that would have saved you a clone ----------------------------
assert_real_filesystem() {
  local target="$1" label="$2"
  local dir; dir="$(dirname "$target")"
  mkdir -p "$dir"
  local probe="$dir/.mmro_symlink_probe.$$"
  if ! ln -s . "$probe" 2>/dev/null; then
    cat >&2 <<EOF

ERROR: $dir does not support symlinks, so a virtualenv cannot be created there.

  This is almost certainly Google Drive. Drive's FUSE mount has no symlinks and
  no working ensurepip, which is exactly how 'python -m venv' fails.

  Fix: point ${label} at LOCAL disk in config.yaml, e.g.

      eval:
        harnesses:
          <harness>:
            venv: /content/MMRM/venvs/<harness>
            repo_dir: /content/MMRM/src/<harness>

  Keep work_dir on Drive so results survive; only the venv and the clone need
  to be local. Re-run with --archive to save the built venvs to Drive so the
  next session restores them instead of reinstalling.

EOF
    return 1
  fi
  rm -f "$probe"
}

archive_path() { echo "${ARCHIVE_DIR}/$1.tar"; }

restore_venv() {
  local name="$1" venv="$2" tar_path
  tar_path="$(archive_path "$name")"
  [[ "$REBUILD" -eq 1 ]] && return 1
  [[ -f "$tar_path" ]] || return 1
  echo "restoring venv from $tar_path"
  mkdir -p "$(dirname "$venv")"
  # Venvs are not relocatable, so this only works because the path is the same
  # absolute path it was built at. That is why venv paths are pinned in config
  # rather than derived from the repo location.
  tar -xf "$tar_path" -C /
  if "$venv/bin/python" --version >/dev/null 2>&1; then
    echo "restored: $("$venv/bin/python" --version)"
    return 0
  fi
  echo "archive did not restore to a working venv; rebuilding"
  rm -rf "$venv"
  return 1
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

  assert_real_filesystem "$venv" "eval.harnesses.${name}.venv"
  assert_real_filesystem "$repo_dir" "eval.harnesses.${name}.repo_dir"

  mkdir -p "$(dirname "$repo_dir")"
  if [[ -d "$repo_dir/.git" ]]; then
    echo "repo already cloned: $repo_dir"
  else
    echo "cloning $repo -> $repo_dir"
    git clone --depth 1 "$repo" "$repo_dir"
  fi
  # Pin what we actually ran. Harness CLIs and task definitions change often, and
  # an unpinned re-clone months later will not reproduce these scores.
  mkdir -p "${REPO_ROOT}/results"
  git -C "$repo_dir" rev-parse HEAD > "${REPO_ROOT}/results/${name}_commit.txt" 2>/dev/null || true

  if [[ -x "$venv/bin/python" ]]; then
    echo "venv already present: $venv"
  elif restore_venv "$name" "$venv"; then
    :
  else
    echo "creating venv: $venv"
    python -m venv "$venv"
    "$venv/bin/python" -m pip install --quiet --upgrade pip

    # Full isolation on purpose: --system-site-packages would let the harness
    # inherit whatever transformers the base environment happens to have, and
    # the gate is about reproducing a published number under the harness's own
    # pins. Correctness beats the ~5GB and ~10 minutes it costs.
    echo "installing ${name} (pulls its own torch; several GB, expect ~10 min)"
    "$venv/bin/python" -m pip install -e "$repo_dir"
    if [[ -f "$reqs" ]]; then
      "$venv/bin/python" -m pip install -r "$reqs"
    fi
  fi

  echo "ready: $("$venv/bin/python" --version) at $venv"

  if [[ "$DO_ARCHIVE" -eq 1 ]]; then
    local tar_path; tar_path="$(archive_path "$name")"
    mkdir -p "$ARCHIVE_DIR"
    echo "archiving venv -> $tar_path (a few GB across Drive; be patient)"
    tar -cf "$tar_path" "$venv"
    echo "archived. Next session restores from here instead of reinstalling."
  fi
}

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
echo "  2. Verify eval.tasks[].baseline against the official model card."
echo "  3. Set eval.judge.* -- published MathVista/MMMU numbers use LLM answer"
echo "     extraction, and rule-based fallback under-scores past the 2pt gate."
echo "  4. Shake out on a subset first:"
echo "       python -m mmro.eval --set eval.enabled=true --set eval.limit=25"
echo "  5. Then the real run: bash scripts/04_eval.sh"
