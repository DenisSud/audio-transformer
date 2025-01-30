from .data_processor import CommonVoiceProcessor
from torch.utils.data import Dataset
import torch
from pathlib import Path

# utils/data.py
class AudioDataset(Dataset):
    def __init__(self, data_path, split, config, max_samples=None):
        """
        Dataset for Common Voice audio clips using preprocessed cache.
        """
        self.config = config

        # Load preprocessed data
        cache_dir = Path(data_path) / 'preprocessed_cache'
        cache_file = cache_dir / f"{split}_data.npz"
        if not cache_file.exists():
            raise FileNotFoundError(
                f"Preprocessed data not found at {cache_file}. "
                "Please run preprocess_dataset.py first with command:\n"
                f"python preprocess_dataset.py --config <config_path>"
            )

        # Load cached data
        cached_data = np.load(cache_file)
        self.audio_data = cached_data['audio']
        self.labels = cached_data['labels']
        self.label_map = cached_data['accent_map'].item()

        # Apply max_samples if specified
        if max_samples and max_samples < len(self.audio_data):
            indices = np.random.choice(len(self.audio_data), max_samples, replace=False)
            self.audio_data = self.audio_data[indices]
            self.labels = self.labels[indices]

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.audio_data[idx]), torch.tensor(self.labels[idx])
