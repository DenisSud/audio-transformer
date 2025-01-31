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
    'data': {
        'root': '/kaggle/input/common-voice'
    },
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
        'num_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'epochs': 30,
        'max_samples': None
    }
}

class AudioDataset(Dataset):
    """Dataset for audio classification with optimized preprocessing."""

    def __init__(self, data_path, split='train', max_samples=None, cache_spectrograms=True):
        self.data_path = Path(data_path)
        self.config = CONFIG['audio']
        self.cache_spectrograms = cache_spectrograms

        # Load metadata
        df = pd.read_csv(self.data_path / f"cv-valid-{split}.csv")

        # Filter by votes
        print(f"Initial samples: {len(df)}")
        df = df[
            (df['up_votes'] > 0) &
            (df['up_votes'] > df['down_votes'])
        ]

        # Clean the accent column
        df = df.dropna(subset=['accent'])
        df['accent'] = df['accent'].astype(str)

        # Filter classes with too few samples
        class_counts = df['accent'].value_counts()
        min_samples_per_class = 10
        valid_classes = class_counts[class_counts >= min_samples_per_class].index
        df = df[df['accent'].isin(valid_classes)]

        if max_samples:
            df = df.sample(n=min(len(df), max_samples))

        # Create label mapping
        self.classes = sorted(df['accent'].unique())
        self.label_map = {label: i for i, label in enumerate(self.classes)}

        # Store file paths and labels
        self.files = df['filename'].values
        self.labels = [self.label_map[accent] for accent in df['accent'].values]

        # Initialize mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config['sr'],
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length'],
            n_mels=self.config['n_mels']
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Cache for spectrograms
        self.spec_cache = {}
        if cache_spectrograms:
            print("Pre-computing mel spectrograms...")
            for idx in tqdm(range(len(self))):
                self._load_and_process_audio(idx)

        print(f"\nDataset ready with {len(self.files)} samples and {len(self.classes)} classes")

    def _load_and_process_audio(self, idx):
        if self.cache_spectrograms and idx in self.spec_cache:
            return self.spec_cache[idx]

        audio_path = self.data_path / "cv-valid-train" / self.files[idx]
        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != self.config['sr']:
                resampler = torchaudio.transforms.Resample(sample_rate, self.config['sr'])
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            target_length = int(self.config['sr'] * self.config['clip_duration'])
            if waveform.shape[1] > target_length:
                waveform = waveform[:, :target_length]
            else:
                waveform = torch.nn.functional.pad(
                    waveform, (0, target_length - waveform.shape[1])
                )

            mel_spec = self.mel_transform(waveform)
            mel_spec = self.amplitude_to_db(mel_spec)
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            mel_spec = torch.zeros((self.config['n_mels'],
                                  int(self.config['sr'] * self.config['clip_duration']
                                      // self.config['hop_length'])))

        if self.cache_spectrograms:
            self.spec_cache[idx] = mel_spec

        return mel_spec

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mel_spec = self._load_and_process_audio(idx)
        return mel_spec, torch.tensor(self.labels[idx])

class AudioTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.config = CONFIG['model']

        # Calculate input dimensions
        self.n_mels = CONFIG['audio']['n_mels']
        sr = CONFIG['audio']['sr']
        duration = CONFIG['audio']['clip_duration']
        hop_length = CONFIG['audio']['hop_length']

        # Calculate the sequence length after mel spectrogram
        self.time_dim = int(duration * sr / hop_length)

        # Conv layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Calculate dimensions after conv layers
        self.freq_dim = self.n_mels // 4
        self.time_dim = self.time_dim // 4

        # Project to embedding dimension
        self.projection = nn.Linear(64 * self.freq_dim, self.config['embed_dim'])

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

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.config['embed_dim'], self.config['embed_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(self.config['dropout']),
            nn.Linear(self.config['embed_dim'] // 2, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)

        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, self.time_dim, 64 * self.freq_dim)

        x = self.projection(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)

        return x

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({
            'loss': f'{total_loss/len(pbar):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
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

def train_model(data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets with caching
    train_dataset = AudioDataset(data_path, split='train', cache_spectrograms=True)
    valid_dataset = AudioDataset(data_path, split='dev', cache_spectrograms=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CONFIG['training']['batch_size'],
        num_workers=4,
        pin_memory=True
    )

    model = AudioTransformer(num_classes=len(train_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['training']['learning_rate']
    )

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
