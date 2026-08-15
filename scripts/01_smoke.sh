#!/usr/bin/env bash
# Smoke-test every enabled model, ONE MODEL PER PROCESS.
#
# The subprocess boundary is load-bearing: two 16.5GB models will not coexist in
# one process on a 24GB L4, and empty_cache() does not reliably return
# fragmented VRAM. Do not "optimize" this into a single python invocation.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

# Fail early and clearly if the held-out images are not in place.
if ! python -m mmro.images >/dev/null 2>&1; then
  python -m mmro.images
  exit 1
fi

models="$(python - <<'PY'
from mmro.config import load_config
print(" ".join(load_config().enabled_models))
PY
)"

if [[ -z "${models// }" ]]; then
  echo "no enabled models in config.yaml"
  exit 1
fi

status=0
for model in $models; do
  echo
  echo "############################################################"
  echo "# smoke: ${model}"
  echo "############################################################"
  python -m mmro.smoke --model "$model" "$@" || status=1
done

echo
echo "############################################################"
python -m mmro.report
exit "$status"
