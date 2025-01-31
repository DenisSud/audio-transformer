"""
experiment.py - Self-contained PyTorch training script for audio classification
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import librosa
import logging

# Configuration
CONFIG = {
    'audio': {
        'sr': 16000,
        'clip_duration': 3.0,
        'n_mels': 80,
        'n_fft': 400,
        'hop_length': 160
    },
    'model': {
        'embed_dim': 512,
        'num_heads': 8,
        'num_layers': 12,
        'dim_feedforward': 2048,
        'dropout': 0.1
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'epochs': 30,
        'max_samples': None  # Set to int to limit dataset size
    }
}

class AudioDataset(Dataset):
    """Dataset for audio classification."""

    def __init__(self, data_path, split='train', max_samples=None):
        self.data_path = Path(data_path)
        self.config = CONFIG['audio']

        # Load metadata
        df = pd.read_csv(self.data_path / f"{split}.csv")
        if max_samples:
            df = df.sample(n=min(len(df), max_samples))

        # Create label mapping
        self.classes = sorted(df['label'].unique())
        self.label_map = {label: i for i, label in enumerate(self.classes)}

        self.files = df['path'].values
        self.labels = [self.label_map[label] for label in df['label'].values]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load and process audio
        audio_path = self.data_path / self.files[idx]
        audio, sr = librosa.load(audio_path, sr=self.config['sr'])

        # Ensure fixed length
        target_length = int(self.config['sr'] * self.config['clip_duration'])
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)))

        # Convert to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config['sr'],
            n_mels=self.config['n_mels'],
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length']
        )

        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        return torch.FloatTensor(mel_spec), torch.tensor(self.labels[idx])

class AudioTransformer(nn.Module):
    """Transformer-based audio classification model."""

    def __init__(self, num_classes):
        super().__init__()

        self.config = CONFIG['model']

        # Conv layers to process mel spectrograms
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Calculate sequence length after convolutions
        self.seq_len = CONFIG['audio']['clip_duration'] * CONFIG['audio']['sr'] // \
                      CONFIG['audio']['hop_length'] // 4  # After two max pools

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config['embed_dim'],
            nhead=self.config['num_heads'],
            dim_feedforward=self.config['dim_feedforward'],
            dropout=self.config['dropout'],
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config['num_layers']
        )

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.config['embed_dim'], self.config['embed_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(self.config['dropout']),
            nn.Linear(self.config['embed_dim'] // 2, num_classes)
        )

    def forward(self, x):
        # Add channel dimension for conv layers
        x = x.unsqueeze(1)

        # Process through conv layers
        x = self.conv_layers(x)

        # Reshape for transformer
        batch_size = x.size(0)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.config['embed_dim'])

        # Transformer processing
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification
        x = self.classifier(x)

        return x

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for inputs, targets in tqdm(loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), correct / total

def plot_metrics(train_losses, val_losses, val_accs):
    """Plot training metrics."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def train_model(data_path, model=None):
    """Main training function."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loaders
    train_dataset = AudioDataset(
        data_path=data_path,
        split='train',
        max_samples=CONFIG['training']['max_samples']
    )

    valid_dataset = AudioDataset(
        data_path=data_path,
        split='valid'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['training']['batch_size'],
        shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CONFIG['training']['batch_size']
    )

    # Model
    if model is None:
        model = AudioTransformer(num_classes=len(train_dataset.classes))
    model = model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['training']['learning_rate']
    )

    # Training loop
    train_losses = []
    val_losses = []
    val_accs = []
    best_acc = 0

    for epoch in range(CONFIG['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['training']['epochs']}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

    plot_metrics(train_losses, val_losses, val_accs)

    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_acc': best_acc
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to data directory")
    args = parser.parse_args()

    model, history = train_model(args.data_path)
