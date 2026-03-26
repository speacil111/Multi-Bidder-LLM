#!/usr/bin/env bash
set -euo pipefail

# 默认参数，可按需改  
TOPK_LIST="${TOPK_LIST:-20,50,}"
MULTIPLIER_LIST="${MULTIPLIER_LIST:-1.5,2.0}"
PPL_TOLERANCE="${PPL_TOLERANCE:-0.25}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-80}"
HIT_KEYWORDS="${HIT_KEYWORDS:-Hilton}"
OUTPUT_DIR="${OUTPUT_DIR:-hilton_topk_search_output}"
ATTR_CACHE="${ATTR_CACHE:-attr_score_cache/attribution_Hilton_Hotel_ig20_3a424a103eeef02c.pt}"

SCRIPT_PATH="find_best_hilton_neuron.py"

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "未找到脚本: ${SCRIPT_PATH}"
  exit 1
fi

if [[ ! -f "${ATTR_CACHE}" ]]; then
  echo "未找到归因缓存: ${ATTR_CACHE}"
  exit 1
fi

echo "开始运行 Hilton neuron 网格搜索..."
echo "TOPK_LIST=${TOPK_LIST}"
echo "MULTIPLIER_LIST=${MULTIPLIER_LIST}"
echo "PPL_TOLERANCE=${PPL_TOLERANCE}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "HIT_KEYWORDS=${HIT_KEYWORDS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "ATTR_CACHE=${ATTR_CACHE}"

python "${SCRIPT_PATH}" \
  --topk-list "${TOPK_LIST}" \
  --multiplier-list "${MULTIPLIER_LIST}" \
  --ppl-tolerance "${PPL_TOLERANCE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --hit-keywords "${HIT_KEYWORDS}" \
  --output-dir "${OUTPUT_DIR}" \
  --attribution-cache "${ATTR_CACHE}" \
  "$@"

echo "运行完成。请查看 ${OUTPUT_DIR} 下的 CSV 和 hilton_combo_outputs_*.txt"
