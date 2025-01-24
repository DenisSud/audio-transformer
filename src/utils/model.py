import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from typing import Optional

class AudioEmbedder:
    """Handles audio embedding using pretrained Wav2Vec2 model.

    Args:
        config (dict): Configuration dictionary containing model parameters
    """
    def __init__(self, config: dict):
        self.config = config
        self.device = config["training"]["device"]
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.model.eval().to(self.device)

        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def get_embeddings(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from audio input."""
        with torch.no_grad():
            inputs = self.processor(
                audio,
                sampling_rate=self.config["audio"]["sr"],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            outputs = self.model(**inputs).last_hidden_state
            return outputs.mean(dim=1)

class ClipTransformer(nn.Module):
    """Transformer model for audio clip classification.

    Args:
        num_classes (int): Number of output classes
        config (dict): Configuration dictionary containing model parameters
    """
    def __init__(self, num_classes: int, config: dict):
        super().__init__()
        self.config = config

        # Positional encoding
        self.pos_encoder = nn.Embedding(
            config["audio"]["max_clips"],
            config["model"]["embed_dim"]
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["model"]["embed_dim"],
            nhead=config["model"]["nhead"],
            dim_feedforward=config["model"]["dim_feedforward"],
            dropout=config["model"]["dropout"],
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            config["model"]["transformer_layers"]
        )

        # Classification head
        self.classifier = nn.Linear(config["model"]["embed_dim"], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        # Add positional encodings
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = x + self.pos_encoder(positions)

        # Apply transformer
        x = self.transformer(x)

        # Pool and classify
        x = x.mean(dim=1)
        return self.classifier(x)

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: Optional[str] = None) -> None:
        """Load model weights."""
        state_dict = torch.load(path, map_location=device if device else self.config["training"]["device"])
        self.load_state_dict(state_dict)
