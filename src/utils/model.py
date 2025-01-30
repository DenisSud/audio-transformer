import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from typing import Optional

class AudioTransformer(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        
        # Mel spectrogram parameters
        self.n_mels = 80
        self.n_fft = 400
        self.hop_length = 160
        
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
        self.seq_len = config['audio']['sr'] * config['audio']['clip_duration'] // self.hop_length // 2
        
        # Project to transformer dimension
        self.projection = nn.Linear(64 * (self.n_mels // 2), config['model']['embed_dim'])
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.seq_len, config['model']['embed_dim'])
        )
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config['model']['embed_dim'],
                nhead=config['model']['nhead'],
                dim_feedforward=config['model']['dim_feedforward'],
                dropout=config['model']['dropout'],
                batch_first=True
            ),
            num_layers=config['model']['transformer_layers']
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config['model']['embed_dim'], config['model']['embed_dim']),
            nn.LayerNorm(config['model']['embed_dim']),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(config['model']['embed_dim'], num_classes)
        )
        
    def forward(self, x):
        # Convert to mel spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )(x)
        
        # Add channel dimension and process through conv layers
        x = mel_spec.unsqueeze(1)
        x = self.conv_layers(x)
        
        # Reshape for transformer
        batch_size = x.size(0)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, self.seq_len, -1)
        
        # Project to transformer dimension
        x = self.projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Transformer processing
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        return self.classifier(x)

class AudioEmbedder:
    """Handles audio embedding using pretrained Wav2Vec2 model.

    Args:
        config (dict): Configuration dictionary containing model parameters
    """
    def __init__(self, config):
        self.config = config
        self.device = config["training"]["device"]
        
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.model.eval().to(self.device)
        
        # Add temporal pooling layer
        self.temporal_pool = nn.Sequential(
            nn.Conv1d(self.model.config.hidden_size, config["model"]["embed_dim"], 1),
            nn.AdaptiveAvgPool1d(1)
        ).to(self.device)
        
        for param in self.model.parameters():
            param.requires_grad = False
            
    def get_embeddings(self, audio):
        with torch.no_grad():
            inputs = self.processor(
                audio,
                sampling_rate=self.config["audio"]["sr"],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            outputs = self.model(**inputs).last_hidden_state
            
            # Apply temporal pooling
            outputs = outputs.transpose(1, 2)
            outputs = self.temporal_pool(outputs)
            outputs = outputs.squeeze(-1)
            
            return outputs

class ClipTransformer(nn.Module):
    """Transformer model for audio clip classification.

    Args:
        num_classes (int): Number of output classes
        config (dict): Configuration dictionary containing model parameters
    """
    def __init__(self, num_classes, config):
        super().__init__()
        self.config = config
        
        # Local attention for processing clip-level features
        self.local_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config["model"]["embed_dim"],
                nhead=config["model"]["nhead"],
                dim_feedforward=config["model"]["dim_feedforward"],
                dropout=config["model"]["dropout"],
                batch_first=True
            ),
            num_layers=2  # Fewer layers for local processing
        )
        
        # Global attention for sequence-level understanding
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config["model"]["embed_dim"],
                nhead=config["model"]["nhead"],
                dim_feedforward=config["model"]["dim_feedforward"],
                dropout=config["model"]["dropout"],
                batch_first=True
            ),
            num_layers=config["model"]["transformer_layers"]
        )
        
        # Learnable CLS token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["model"]["embed_dim"]))
        
        # Relative positional encoding
        self.relative_pos_encoder = nn.Parameter(
            torch.randn(2 * config["audio"]["max_clips"] - 1, config["model"]["embed_dim"])
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(config["model"]["embed_dim"], config["model"]["embed_dim"]),
            nn.LayerNorm(config["model"]["embed_dim"]),
            nn.ReLU(),
            nn.Dropout(config["model"]["dropout"]),
            nn.Linear(config["model"]["embed_dim"], num_classes)
        )

    def hello(messag: str):
        return messga + "hello"

    def forward(self, x):
        batch_size = x.size(0)
        
        # Process each clip with local attention
        x = self.local_transformer(x)
        
        # Add CLS token for global representation
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply global transformer
        x = self.global_transformer(x)
        
        # Use CLS token for classification
        x = x[:, 0]  # Take CLS token representation
        
        return self.classifier(x)
