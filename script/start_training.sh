#!/bin/bash

# This script starts the training process for the WiFi Sensing project.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Single GPU Training ---
# We will use the default GPU (usually 0). If you want to specify one, uncomment the next line.
# export CUDA_VISIBLE_DEVICES=0 

echo "Starting training on a single GPU."

# Project root directory is one level up from the script directory
PROJECT_ROOT=$(dirname "$0")/..
TRAIN_SCRIPT="$PROJECT_ROOT/src/train.py"
CONFIG_FILE="$PROJECT_ROOT/config.yaml"

# Use a standard python command to launch the training
python "$TRAIN_SCRIPT" --config "$CONFIG_FILE"

echo "Training finished." 