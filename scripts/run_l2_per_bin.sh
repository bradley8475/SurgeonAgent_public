#!/usr/bin/env bash
# 按 bin 起 4 个 L2 进程：easy/medium/hard 各 5 个，very_hard 2 个，共 17 样本。
set -u
cd "$(dirname "$0")/.."

export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:-}/opt/homebrew/lib"

STAMP=$(date +%Y%m%d_%H%M%S)
LAUNCH_LOG_DIR="logs/l2_per_bin_launch/${STAMP}"
mkdir -p "${LAUNCH_LOG_DIR}"

EASY="f1f6adff3f0ee88d,1289d8499ac699e9,567773ca056280d2,2b1cd0b309671cec,dd129fd70b247c61"
MEDIUM="rand_CYT17KC43E,6fa7c18c0cb39f3b,rand_BTK1Z0PPLG,rand_ATDY3AJN0Q,4c71a170ee4de882"
HARD="rand_00YONAPXZE,rand_249RUXCM75,rand_4FIT41GWOP,rand_EDOV8CMMAJ,rand_86FF7HFBGU"
VERY_HARD="rand_4M2IM1ZCSG,rand_BH4KRHNR47"

declare -a TAGS=("easy" "medium" "hard" "very_hard")
declare -a IDLISTS=("${EASY}" "${MEDIUM}" "${HARD}" "${VERY_HARD}")

PIDS=()
for i in "${!TAGS[@]}"; do
  TAG="${TAGS[$i]}"
  IDS="${IDLISTS[$i]}"
  N=$(awk -F, '{print NF}' <<<"${IDS}")
  OUT="${LAUNCH_LOG_DIR}/${TAG}.log"
  echo "[launch] tag=${TAG} n=${N} -> ${OUT}"
  uv run python utils/eval/run_l2.py --samples "${IDS}" --tag "${TAG}" \
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
