#!/usr/bin/env bash
# Train ByteTabNet (PyTorch) on the Quora Question-Answer dataset
# Dataset: https://huggingface.co/datasets/toughdata/quora-question-answer-dataset
#
# Usage:
#   chmod +x run_quora_qa.sh
#   ./run_quora_qa.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/trainer.py" --config "$SCRIPT_DIR/../configs/quora_qa_generation.json" --reverse-kl-weight 0.1
