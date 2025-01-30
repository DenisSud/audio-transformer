import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from typing import Optional

class AudioTransformer(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()

        # Mel spectrogram parameters
        self.n_mels = 80
        self.n_fft = 400
        self.hop_length = 160

        # Initialize mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config['audio']['sr'],
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        # Convert embed_dim to int
        embed_dim = int(config['model']['embed_dim'])

        # Convolutional front-end to process mel spectrograms
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Calculate sequence length after conv layers
        self.seq_len = int(config['audio']['sr'] * config['audio']['clip_duration'] // self.hop_length // 2)

        # Project to transformer dimension
        self.projection = nn.Sequential(
            nn.Linear(64 * (self.n_mels // 2), embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.seq_len, embed_dim)
        )

        # Transformer layers with residual connections
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=config['model']['nhead'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout'],
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['model']['transformer_layers']
        )

        # Classification head with residual connection
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(embed_dim, embed_dim),  # Residual connection will be added in forward pass
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(x)

        # Add channel dimension and process through conv layers
        x = mel_spec.unsqueeze(1)
        x = self.conv_layers(x)

        # Reshape for transformer
        batch_size = x.size(0)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, self.seq_len, -1)

        # Project to transformer dimension
        proj = self.projection(x)

        # Add positional encoding with residual connection
        x = proj + self.pos_encoding

        # Transformer processing (residual connections are handled inside transformer)
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification with residual connection
        x_res = x
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if i == 4:  # Add residual connection after second linear layer
                x = x + x_res

        return x
