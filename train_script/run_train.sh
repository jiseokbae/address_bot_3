#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_PATH=${1:-"$SCRIPT_DIR/config/train_klue_1000_3_3_2.yaml"}

python "$SCRIPT_DIR/train_bert.py" --config "$CONFIG_PATH"