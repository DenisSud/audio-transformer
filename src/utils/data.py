from .data_processor import CommonVoiceProcessor
from .model import ClipTransformer, AudioEmbedder
from torch.utils.data import Dataset
import torch

class AudioClipDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 split: str,
                 config: dict,
                 max_samples: int = None):
        """
        Dataset for Common Voice audio clips.

        Args:
            data_path: Path to data directory
            split: Dataset split ('train' or 'dev')
            config: Configuration dictionary
            max_samples: Maximum number of samples to use
        """
        self.config = config
        self.processor = CommonVoiceProcessor(config)
        self.embedder = AudioEmbedder(config)

        # Load dataset
        self.audio_paths, self.labels, self.label_map = \
            self.processor.prepare_dataset(data_path, split, max_samples)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        # Load and process audio
        audio = self.processor.process_audio(self.audio_paths[idx])

        # Get embeddings
        embeddings = self.embedder.get_embeddings(audio)

        return embeddings, torch.tensor(self.labels[idx])
