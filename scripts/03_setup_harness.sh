#!/usr/bin/env bash
# Stage 0.2 — build the two benchmark harnesses in ISOLATED venvs.
#
# VLMEvalKit and lmms-eval pin conflicting dependencies, and both would clobber
# the torch build that stage 0 runs on. Each gets its own venv with its own
# torch; the base environment is never touched. The backends shell out, so the
# pins never meet.
#
# Two separate things break venv creation here, and they look identical:
#
#   1. LOCATION. A virtualenv needs symlinks; Google Drive's FUSE mount has
#      none. A git clone succeeds there because it only writes regular files,
#      so the failure surfaces halfway through setup. `assert_real_filesystem`
#      probes for this before anything is installed.
#   2. ENVIRONMENT. Even on local disk, Colab's Python can fail `ensurepip`
#      with a bare "exit status 1" that says nothing about why -- usually pip
#      configuration leaking in from the parent environment. `create_venv`
#      neutralises that, tries three strategies, and prints the real error if
#      all of them fail.
#
# Fixing (1) does not fix (2). Both are handled.
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

# --- venv creation ----------------------------------------------------------
# `python -m venv` fails on Colab often enough that one strategy is not enough.
# The usual causes are environment rather than disk: Colab exports pip settings
# (PIP_TARGET / PYTHONUSERBASE) and ships /etc/pip.conf, and ensurepip inherits
# all of it, then fails with a bare "exit status 1" that says nothing.
#
# So: neutralise the pip environment, then try three ways and report the real
# error if every one fails.
create_venv() {
  local venv="$1" py="${2:-}"
  if [[ -z "$py" ]]; then py=python; fi

  # Colab's pip configuration must not reach the bootstrap.
  local -a clean=(
    env -u PIP_TARGET -u PIP_PREFIX -u PIP_USER -u PYTHONUSERBASE
        -u PYTHONHOME -u PYTHONPATH
    PIP_CONFIG_FILE=/dev/null
  )

  echo "creating venv: $venv"

  # 1. Standard venv, with the environment cleaned.
  if "${clean[@]}" "$py" -m venv "$venv" 2>"$venv.err"; then
    rm -f "$venv.err"; return 0
  fi
  echo "  python -m venv failed; trying virtualenv"
  rm -rf "$venv"

  # 2. virtualenv bundles its own pip wheels and never calls ensurepip, which
  #    sidesteps the failure entirely. It is a small pure-Python install.
  if ! "$py" -c "import virtualenv" >/dev/null 2>&1; then
    "${clean[@]}" "$py" -m pip install --quiet virtualenv || true
  fi
  if "${clean[@]}" "$py" -m virtualenv "$venv" 2>>"$venv.err"; then
    rm -f "$venv.err"; return 0
  fi
  echo "  virtualenv failed; trying venv --without-pip + get-pip.py"
  rm -rf "$venv"

  # 3. Build the venv with no pip at all, then bootstrap pip into it by hand.
  if "${clean[@]}" "$py" -m venv --without-pip "$venv" 2>>"$venv.err"; then
    local getpip="${venv}.get-pip.py"
    if curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$getpip" \
       && "${clean[@]}" "$venv/bin/python" "$getpip" --quiet; then
      rm -f "$getpip" "$venv.err"; return 0
    fi
    rm -f "$getpip"
  fi

  cat >&2 <<EOF

ERROR: could not create a virtualenv at $venv. All three strategies failed.

  The underlying error (usually hidden behind "exit status 1"):
EOF
  sed 's/^/    /' "$venv.err" >&2 2>/dev/null || true
  cat >&2 <<EOF

  Reproduce it directly to see the full trace:

      python -m venv /tmp/probe_venv          # the failure
      python -m ensurepip --version           # is ensurepip present at all?
      python -c "import sys; print(sys.prefix, sys.base_prefix)"
      env | grep -iE '^(PIP_|PYTHON)'         # Colab's pip environment
      cat /etc/pip.conf 2>/dev/null

  Common fixes:
    * apt-get install -y python3-venv        (ensurepip missing entirely)
    * pip install virtualenv                 (then re-run this script)
    * unset PIP_TARGET                       (Colab sometimes exports it)

EOF
  return 1
}

# --- dependency doctor ------------------------------------------------------
# Both harnesses import large registries at package-import time, so ONE missing
# optional dependency stops the run before it reaches your task. Discovering
# them one benchmark launch at a time is slow, so this loop imports, installs
# whatever is missing, and repeats -- then records what it added so the next
# environment does not have to rediscover it.
#
# Import name != package name for a handful of these, hence the mapping.
pypi_name() {
  case "$1" in
    cv2)         echo "opencv-python-headless" ;;
    PIL)         echo "pillow" ;;
    sklearn)     echo "scikit-learn" ;;
    yaml)        echo "pyyaml" ;;
    rouge_score) echo "rouge-score" ;;
    Levenshtein) echo "python-Levenshtein" ;;
    fitz)        echo "pymupdf" ;;
    # decord ships no wheel for Python 3.12+; the fork does.
    decord)      echo "eva-decord" ;;
    *)           echo "$1" ;;
  esac
}

doctor_imports() {
  local name="$1" venv="$2" probe="$3"
  if [[ -z "$probe" ]]; then return 0; fi

  local auto="requirements/harness_${name}.auto.txt"
  local -a installed=()
  local max=15 i err missing pkg

  echo "verifying imports: $probe"
  for ((i = 0; i < max; i++)); do
    if "$venv/bin/python" -c "$probe" 2>/tmp/mmro_probe.err; then
      if ((${#installed[@]})); then
        printf '%s\n' "${installed[@]}" > "$auto"
        echo "  resolved by installing: ${installed[*]}"
        echo "  recorded in $auto -- promote these into requirements/harness_${name}.txt"
      else
        echo "  all imports satisfied"
      fi
      return 0
    fi

    missing="$(sed -n "s/^ModuleNotFoundError: No module named '\([^']*\)'.*/\1/p" \
               /tmp/mmro_probe.err | tail -1)"
    if [[ -z "$missing" ]]; then
      echo "  import failed for a reason that is not a missing module:" >&2
      sed 's/^/    /' /tmp/mmro_probe.err >&2
      return 1
    fi

    pkg="$(pypi_name "$missing")"
    echo "  missing '$missing' -> installing $pkg"
    if ! "$venv/bin/python" -m pip install --quiet "$pkg"; then
      cat >&2 <<EOF

ERROR: could not install '$pkg' (needed for '$missing') into $venv.

  If this is a wheel-availability problem, the usual cause is the interpreter
  version. Check it:

      $venv/bin/python --version

  Then pin an older one in config.yaml and rebuild:

      eval:
        harnesses:
          ${name}:
            python_bin: python3.11

      bash scripts/03_setup_harness.sh --rebuild

EOF
      return 1
    fi
    installed+=("$pkg")
  done

  echo "  gave up after $max missing modules; something larger is wrong" >&2
  return 1
}

restore_venv() {
  local name="$1" venv="$2" tar_path
  tar_path="$(archive_path "$name")"
  if [[ "$REBUILD" -eq 1 ]]; then return 1; fi
  if [[ ! -f "$tar_path" ]]; then return 1; fi
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
  local venv repo repo_dir reqs probe python_bin
  venv="$(read_cfg "${name}.venv")"
  repo="$(read_cfg "${name}.repo")"
  repo_dir="$(read_cfg "${name}.repo_dir")"
  probe="$(read_cfg "${name}.import_probe")"
  python_bin="$(read_cfg "${name}.python_bin")"
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
    create_venv "$venv" "$python_bin" || return 1
    "$venv/bin/python" -m pip install --quiet --upgrade pip

    # Full isolation on purpose: --system-site-packages would let the harness
    # inherit whatever transformers the base environment happens to have, and
    # the gate is about reproducing a published number under the harness's own
    # pins. Correctness beats the ~5GB and ~10 minutes it costs.
    echo "installing ${name} (pulls its own torch; several GB, expect ~10 min)"
    "$venv/bin/python" -m pip install -e "$repo_dir"
    # VLMEvalKit ships a requirements.txt that the editable install ignores.
    if [[ -f "$repo_dir/requirements.txt" ]]; then
      "$venv/bin/python" -m pip install -r "$repo_dir/requirements.txt" || true
    fi
    if [[ -f "$reqs" ]]; then
      "$venv/bin/python" -m pip install -r "$reqs"
    fi
  fi

  doctor_imports "$name" "$venv" "$probe" || return 1

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
