#!/usr/bin/env bash

set -euo pipefail

# Directory containing this script and the Python visualizer
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default parameters for the visualization
DATASET_ROOT="/work/vajira/DATA/SPERM_data_2025/manually_annotated_100_frames_v0.1"
SPLIT_SUBDIR="all"
OUTPUT_DIR="${DATASET_ROOT}/vis_all"
MAX_IMAGES=50

echo "Using dataset root: ${DATASET_ROOT}"
echo "Split subdir:       ${SPLIT_SUBDIR}"
echo "Output directory:   ${OUTPUT_DIR}"
echo "Max images:         ${MAX_IMAGES}"
echo

mkdir -p "${OUTPUT_DIR}"

python "${SCRIPT_DIR}/visualize_sperm_bboxes.py" \
  --dataset-root "${DATASET_ROOT}" \
  --split-subdir "${SPLIT_SUBDIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-images "${MAX_IMAGES}" \
  "$@"

