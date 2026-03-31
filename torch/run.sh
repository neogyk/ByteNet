#!/usr/bin/env bash
# Train ByteTabNet (PyTorch) on the ChemistryQA dataset
# Dataset: https://huggingface.co/datasets/avaliev/ChemistryQA
#
# Usage:
#   chmod +x run.sh
#   ./run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/torch"

python3 trainer.py --config "$SCRIPT_DIR/configs/chemistry_qa_generation.json"
