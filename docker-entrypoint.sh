#!/bin/bash
set -e

# Function to display help message
show_help() {
    echo "Usage: docker run [OPTIONS] image [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  train     Train a new model"
    echo "  infer     Run inference on audio files"
    echo "  finetune  Fine-tune an existing model"
    echo ""
    echo "Examples:"
    echo "  docker run audio-classifier train --config config.yml"
    echo "  docker run audio-classifier infer --audio input.wav --model model.pth"
    echo "  docker run audio-classifier finetune --model model.pth --data new_data/"
}

case "$1" in
    train)
        shift
        python src/train.py "$@"
        ;;
    infer)
        shift
        python src/inference.py "$@"
        ;;
    finetune)
        shift
        python src/finetune.py "$@"
        ;;
    --help)
        show_help
        ;;
    *)
        show_help
        exit 1
        ;;
esac
