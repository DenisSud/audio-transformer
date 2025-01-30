import os
import pandas as pd
import librosa
import soundfile as sf
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import numpy as np

class CommonVoiceProcessor:
    """Processor for Common Voice dataset."""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.sample_rate = config["audio"]["sr"]
        self.clip_duration = config["audio"]["clip_duration"]

    def load_data(self, data_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load Common Voice data from TSV files.

        Args:
            data_path: Path to data directory containing TSV files

        Returns:
            Tuple of (processed DataFrame, label mapping dict)
        """
        # Load csv file
        df = pd.read_csv(data_path) 

        # Basic filtering
        df = df[
            (df['up_votes'].fillna(0) >= 2) &  
            df['text'].notna() &  # Has transcription
            df['accent'].notna()  # Has transcription
        ]

        # Create accent mapping
        unique_accents = sorted(df['accent'].unique())
        accent_map = {accent: idx for idx, accent in enumerate(unique_accents)}

        # Add numeric label
        df['label'] = df['accent'].map(accent_map)

        self.logger.info(f"Loaded {len(df)} valid samples with {len(accent_map)} accent classes")
        return df, accent_map

    def process_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess single audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Processed audio array
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        # Apply preprocessing
        audio = self._preprocess_audio(audio)

        # Pad/trim to fixed length
        target_length = int(self.clip_duration * self.sample_rate)
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)))

        return audio

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio preprocessing steps."""
        # Remove silence
        audio = librosa.effects.trim(audio)[0]

        # Normalize
        audio = librosa.util.normalize(audio)

        # Add noise for robustness
        noise_factor = 0.005
        noise = np.random.normal(0, noise_factor, audio.shape)
        audio = audio + noise

        return audio

    def prepare_dataset(self,
                       data_path: str,
                       split: str = 'train',
                       max_samples: int = None) -> Tuple[List[str], List[int]]:
        """
        Prepare dataset for training/validation.

        Args:
            data_path: Path to data directory
            split: Dataset split ('train' or 'dev')
            max_samples: Maximum number of samples to use

        Returns:
            Tuple of (audio file paths, labels)
        """
        # Load and filter data
        df, accent_map = self.load_data(os.path.join(data_path, f'cv-valid-{split}.csv'))

        if max_samples:
            df = df.sample(n=min(max_samples, len(df)))

        # Get file paths and labels
        audio_paths = [os.path.join(data_path, f"cv-valid-{split}", f) for f in df['filename']]
        labels = df['label'].tolist()

        return audio_paths, labels, accent_map
