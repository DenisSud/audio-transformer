import os
import torch
import numpy as np
from tqdm import tqdm
import yaml
from pathlib import Path
from utils.data_processor import CommonVoiceProcessor
import argparse

def preprocess_dataset(config_path: str):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create processor
    processor = CommonVoiceProcessor(config)

    # Create cache directory
    cache_dir = Path(config['dataset']['root']) / 'preprocessed_cache'
    cache_dir.mkdir(exist_ok=True)

    # Process both train and dev splits
    for split in ['train', 'dev']:
        print(f"Processing {split} split...")

        # Get dataset info
        audio_paths, labels, accent_map = processor.prepare_dataset(
            config['dataset']['root'],
            split,
            config['dataset'].get('max_samples')
        )

        # Process and save each audio file
        processed_data = []
        for audio_path in tqdm(audio_paths):
            # Process audio
            processed_audio = processor.process_audio(audio_path)
            processed_data.append(processed_audio)

        # Save processed data and labels
        processed_data = np.array(processed_data)
        cache_file = cache_dir / f"{split}_data.npz"
        np.savez_compressed(
            cache_file,
            audio=processed_data,
            labels=labels,
            accent_map=accent_map
        )

        print(f"Saved preprocessed data to {cache_file}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess audio dataset")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    args = parser.parse_args()

    preprocess_dataset(args.config)

if __name__ == "__main__":
    main()
