#!/usr/bin/env bash
# 高并发全量实验：每个 sample 一个进程，xargs -P 控制并发数。
# 用法:
#   bash scripts/run_all_parallel.sh l2          # L2 with tools (全量 100 样本)
#   bash scripts/run_all_parallel.sh l2-notools  # L2 no-tools ablation
#   bash scripts/run_all_parallel.sh b1          # B1 baseline (mimo)
#   bash scripts/run_all_parallel.sh all         # 依次跑 l2 -> l2-notools -> b1
#
# 可选环境变量:
#   CONCURRENCY=20    并发进程数 (默认 20)
#   SAMPLES=id1,id2   只跑指定样本 (默认全量)
set -u
cd "$(dirname "$0")/.."

export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:-}/opt/homebrew/lib"

CONCURRENCY="${CONCURRENCY:-20}"
OVERALL_MODE="${1:-all}"
export STAMP=$(date +%Y%m%d_%H%M%S)

# 从 test_meta.json 提取全部 sample_id
ALL_IDS=$(python3 -c "
import json, os
with open('splits/test_meta.json') as f:
    meta = json.load(f)
for entry in meta:
    sid = os.path.basename(os.path.dirname(entry['path']))
    print(sid)
")

# 如果指定了 SAMPLES 环境变量，过滤
if [[ -n "${SAMPLES:-}" ]]; then
  FILTER_FILE=$(mktemp)
  echo "${SAMPLES}" | tr ',' '\n' > "${FILTER_FILE}"
  ALL_IDS=$(echo "${ALL_IDS}" | grep -F -f "${FILTER_FILE}")
  rm -f "${FILTER_FILE}"
fi

N_TOTAL=$(echo "${ALL_IDS}" | wc -l | tr -d ' ')
echo "=========================================="
echo " Mode: ${OVERALL_MODE}"
echo " Concurrency: ${CONCURRENCY}"
echo " Total samples: ${N_TOTAL}"
echo " Timestamp: ${STAMP}"
echo "=========================================="

run_batch() {
  local MODE_NAME="$1"
  export MODE="${MODE_NAME}"
  export LOG_DIR="logs/batch_${MODE_NAME}/${STAMP}"
  mkdir -p "${LOG_DIR}"

  echo ""
  echo "[${MODE_NAME}] starting ${N_TOTAL} samples, concurrency=${CONCURRENCY}"
  echo "[${MODE_NAME}] logs -> ${LOG_DIR}/"

  echo "${ALL_IDS}" | xargs -P "${CONCURRENCY}" -I {} bash scripts/_run_one.sh {}

  local FAIL_COUNT=$(grep -rl "Traceback\|unhandled" "${LOG_DIR}"/ 2>/dev/null | wc -l | tr -d ' ')
  echo "[${MODE_NAME}] done. potential failures: ${FAIL_COUNT}/${N_TOTAL}"
}

case "${OVERALL_MODE}" in
  l2)
    run_batch "l2"
    ;;
  l2-notools)
    run_batch "l2-notools"
    ;;
  b1)
    run_batch "b1"
    ;;
  all)
    run_batch "l2"
    echo ""
    echo "========== L2 done, starting L2-notools =========="
    run_batch "l2-notools"
    echo ""
    echo "========== L2-notools done, starting B1 =========="
    run_batch "b1"
    ;;
  *)
    echo "Usage: $0 {l2|l2-notools|b1|all}"
    exit 1
    ;;
esac

echo ""
echo "=========================================="
echo " ALL DONE — ${OVERALL_MODE} @ ${STAMP}"
echo "=========================================="
