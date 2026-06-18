#!/usr/bin/env bash
# Helper: 跑单个 sample. 被 run_all_parallel.sh 通过 xargs 调用。
# 参数: $1=sample_id  环境变量: MODE, LOG_DIR, STAMP
set -u

export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:-}/opt/homebrew/lib"

SID="$1"
LOG_FILE="${LOG_DIR}/${SID}.log"

echo "[start] ${SID}"

case "${MODE}" in
  l2)
    uv run python utils/eval/run_l2.py --samples "${SID}" --tag "${STAMP}_${SID}" \
      >"${LOG_FILE}" 2>&1
    ;;
  l2-notools)
    uv run python utils/eval/run_l2.py --samples "${SID}" --tag "${STAMP}_${SID}" --no-tools \
      >"${LOG_FILE}" 2>&1
    ;;
  b1)
    uv run python utils/eval/run_b1.py --config config/b1_baseline_mimo.yaml \
      --samples "${SID}" --tag "${STAMP}_${SID}" \
      >"${LOG_FILE}" 2>&1
    ;;
esac

STATUS=$?
if [[ ${STATUS} -ne 0 ]]; then
  echo "[FAIL] ${SID} (exit=${STATUS})"
else
  echo "[ ok ] ${SID}"
fi
