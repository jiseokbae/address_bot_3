#!/bin/bash

set -e

CONFIG_PATH=${1:-config/train_klue_1000_3_3_2.yaml}

python train_bert.py --config "$CONFIG_PATH"