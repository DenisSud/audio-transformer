import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
from typing import List, Tuple
from .model import AudioEmbedder

class AudioClipDataset(Dataset):
    """Dataset class for loading and processing audio clips.

    Args:
        file_list (List[str]): List of paths to audio files
        labels (List[int]): List of corresponding labels
        config (dict): Configuration dictionary containing audio parameters
    """
    def __init__(self, file_list: List[str], labels: List[int], config: dict):
        self.file_list = file_list
        self.labels = labels
        self.config = config
        self.embedder = AudioEmbedder(config)
        self.clip_length = int(config["audio"]["sr"] * config["audio"]["clip_duration"])

    def __len__(self) -> int:
        return len(self.file_list)

    def _load_and_process_audio(self, audio_path: str) -> np.ndarray:
        """Load and preprocess audio file."""
        try:
            audio, _ = librosa.load(audio_path, sr=self.config["audio"]["sr"])
            if len(audio) < self.clip_length:
                audio = librosa.util.fix_length(audio, size=self.clip_length)
            return audio
        except Exception as e:
            raise RuntimeError(f"Error loading audio file {audio_path}: {str(e)}")

    def _get_clips(self, audio: np.ndarray) -> np.ndarray:
        """Split audio into fixed-length clips."""
        clips = librosa.util.frame(
            audio,
            frame_length=self.clip_length,
            hop_length=self.clip_length
        ).T
        return clips

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio = self._load_and_process_audio(self.file_list[idx])
        clips = self._get_clips(audio)

        clip_embeddings = []
        for clip in clips:
            if len(clip) < self.clip_length:
                clip = librosa.util.fix_length(clip, size=self.clip_length)
            embeddings = self.embedder.get_embeddings(clip)
            clip_embeddings.append(embeddings.cpu())

        embeddings_tensor = torch.cat(clip_embeddings, dim=0)

        # Pad or truncate to max_clips
        if embeddings_tensor.shape[0] > self.config["audio"]["max_clips"]:
            embeddings_tensor = embeddings_tensor[:self.config["audio"]["max_clips"]]
        else:
            padding = torch.zeros((
                self.config["audio"]["max_clips"] - embeddings_tensor.shape[0],
                self.config["model"]["embed_dim"]
            ))
            embeddings_tensor = torch.cat([embeddings_tensor, padding], dim=0)

        return embeddings_tensor, torch.tensor(self.labels[idx])
