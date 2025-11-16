#!/usr/bin/env bash
# run_seeds.sh
set -euo pipefail

MODE="${1:-tom_coop}"            # usage: ./run_seeds.sh coop
PY=python
SCRIPT="coop2_modular.py"
STEPS=60
SIZE=7
FPS=5
VIZ_FLAG="--viz"             # change to "" to disable visualization for batch runs

mkdir -p _log

for SEED in $(seq 1 10); do
  LOG="_log/${MODE}_${SEED}.txt"
  echo "==> Running mode=${MODE}, seed=${SEED} (logging to ${LOG})"
  ${PY} "${SCRIPT}" ${VIZ_FLAG} \
    --steps "${STEPS}" --size "${SIZE}" --seed "${SEED}" --fps "${FPS}" --mode "${MODE}" \
    > "${LOG}" 2>&1
done

echo "All done. Logs in _log/"
