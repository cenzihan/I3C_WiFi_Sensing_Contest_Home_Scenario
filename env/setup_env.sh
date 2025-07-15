#!/bin/bash
# This script creates a conda environment from the environment.yml file.

echo "Creating conda environment 'wifi_sensing'..."
conda env create -f env/environment.yml
echo ""
echo "Conda environment 'wifi_sensing' created successfully."
echo "To activate the environment, run: conda activate wifi_sensing" 