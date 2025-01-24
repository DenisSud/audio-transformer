#!/bin/bash

# Create necessary directories
mkdir -p data/input data/output models/checkpoints

# Check if NVIDIA Docker is installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: NVIDIA drivers not found. GPU support will not be available."
    echo "Please install NVIDIA drivers and NVIDIA Docker for GPU support."
fi

# Build Docker image
docker-compose build

echo "Setup complete! You can now run:"
echo "docker-compose run audio-classifier --help"
