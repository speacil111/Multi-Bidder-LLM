#!/bin/bash
set -uo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
exec "${PYTHON_BIN}" topk_sweep_inprocess.py "$@"
