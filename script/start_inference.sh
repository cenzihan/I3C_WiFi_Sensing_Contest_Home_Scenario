#!/bin/bash

# This script starts the inference process for the WiFi Sensing project.

# Exit immediately if a command exits with a non-zero status.
set -e

# Project root directory is one level up from the script directory
PROJECT_ROOT=$(dirname "$0")/..
INFERENCE_SCRIPT="$PROJECT_ROOT/src/inference.py"
CONFIG_FILE="$PROJECT_ROOT/config.yaml"

# Read the model path from the config file to inform the user
MODEL_PATH=$(grep 'model_path:' "$CONFIG_FILE" | awk '{print $2}')

echo "Starting inference using model: $MODEL_PATH"

# Use a standard python command to launch the inference
python "$INFERENCE_SCRIPT" --config "$CONFIG_FILE"

echo "Inference finished." 