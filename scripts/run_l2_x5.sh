#!/usr/bin/env bash
# 5 个 L2 进程并发跑（每个进程 1 个 easy 样本，独立 run_dir / log 路径）。
set -u
cd "$(dirname "$0")/.."

export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:-}/opt/homebrew/lib"

STAMP=$(date +%Y%m%d_%H%M%S)
LAUNCH_LOG_DIR="logs/l2_x5_launch/${STAMP}"
mkdir -p "${LAUNCH_LOG_DIR}"

declare -a SAMPLES=(
  "f1f6adff3f0ee88d"
  "1289d8499ac699e9"
  "567773ca056280d2"
  "2b1cd0b309671cec"
  "dd129fd70b247c61"
)

PIDS=()
for i in "${!SAMPLES[@]}"; do
  TAG="p${i}"
  SID="${SAMPLES[$i]}"
  OUT="${LAUNCH_LOG_DIR}/${TAG}_${SID}.log"
  echo "[launch] tag=${TAG} sample=${SID} -> ${OUT}"
  uv run python utils/eval/run_l2.py --samples "${SID}" --tag "${TAG}" \
      >"${OUT}" 2>&1 &
  PIDS+=($!)
done

echo "[launch] launched ${#PIDS[@]} procs: ${PIDS[*]}"
echo "[launch] launch log dir: ${LAUNCH_LOG_DIR}"

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    echo "[launch] pid ${pid} exited non-zero"
    FAIL=$((FAIL+1))
  fi
done

echo "[launch] all done. failed=${FAIL}"
exit "${FAIL}"
