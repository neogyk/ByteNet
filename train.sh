#!/usr/bin/env bash
# Train ByteTabNet (PyTorch) on the Quora Question-Answer dataset
# Dataset: https://huggingface.co/datasets/toughdata/quora-question-answer-dataset
#
# Usage:
#   chmod +x train.sh
#   ./train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/torch"

python trainer.py --config "$SCRIPT_DIR/configs/quora_qa_generation.json"
