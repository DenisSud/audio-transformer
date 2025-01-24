"""
Training script for audio classification model.
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import pandas as pd
from tqdm import tqdm

from utils.data import AudioClipDataset
from utils.model import ClipTransformer
from utils.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    print_classification_report
)

class TrainingManager:
    """Manages the training process for the audio classification model."""

    def __init__(self, config_path: str, experiment_name: Optional[str] = None):
        """
        Initialize training manager.

        Args:
            config_path: Path to configuration file
            experiment_name: Optional name for the experiment
        """
        self.setup_logging()
        self.config = self.load_config(config_path)
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_directories()
        self.setup_device()
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'best_acc': 0.0
        }

    def setup_logging(self) -> None:
        """Configure logging settings."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('training.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load and validate configuration file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self._validate_config(config)
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config: {str(e)}")
            sys.exit(1)

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters."""
        required_keys = ['dataset', 'audio', 'model', 'training', 'output']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

    def setup_directories(self) -> None:
        """Create necessary directories for outputs."""
        self.output_dir = Path(self.config['output']['model_dir']) / self.experiment_name
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.visualization_dir = self.output_dir / "visualizations"

        for directory in [self.output_dir, self.checkpoint_dir, self.visualization_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def setup_device(self) -> None:
        """Setup compute device and distribute across GPUs if available."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.n_gpu = torch.cuda.device_count()
            self.logger.info(f"Using {self.n_gpu} GPU(s)")
        else:
            self.device = torch.device("cpu")
            self.n_gpu = 0
            self.logger.info("Using CPU")

    def prepare_data(self) -> tuple:
        """
        Prepare training and validation datasets.

        Returns:
            Tuple of (train_loader, valid_loader, label_map)
        """
        try:
            # Load and preprocess data
            train_df = pd.read_csv(Path(self.config['dataset']['root']) / "cv-valid-train.csv")
            valid_df = pd.read_csv(Path(self.config['dataset']['root']) / "cv-valid-dev.csv")

            # Filter valid samples
            train_df = train_df[
                (train_df['accent'].notna()) &
                (train_df['up_votes'] > train_df['down_votes'])
            ]
            valid_df = valid_df[
                (valid_df['accent'].notna()) &
                (valid_df['up_votes'] > valid_df['down_votes'])
            ]

            # Create label mapping
            all_accents = pd.concat([train_df['accent'], valid_df['accent']]).unique()
            label_map = {accent: idx for idx, accent in enumerate(sorted(all_accents))}

            # Save label mapping
            with open(self.output_dir / "label_map.yaml", 'w') as f:
                yaml.dump(label_map, f)

            # Create datasets
            train_dataset = AudioClipDataset(
                file_list=[Path(self.config['dataset']['root']) / "cv-valid-train" / f for f in train_df['filename']],
                labels=train_df['accent'].map(label_map).values,
                config=self.config
            )

            valid_dataset = AudioClipDataset(
                file_list=[Path(self.config['dataset']['root']) / "cv-valid-dev" / f for f in valid_df['filename']],
                labels=valid_df['accent'].map(label_map).values,
                config=self.config
            )

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=min(os.cpu_count(), 4),
                pin_memory=True
            )

            valid_loader = DataLoader(
                valid_dataset,
                batch_size=self.config['training']['batch_size'],
                num_workers=min(os.cpu_count(), 4),
                pin_memory=True
            )

            return train_loader, valid_loader, label_map

        except Exception as e:
            self.logger.error(f"Failed to prepare data: {str(e)}")
            raise

    def train(self) -> None:
        """Main training loop."""
        try:
            train_loader, valid_loader, label_map = self.prepare_data()

            # Initialize model
            model = ClipTransformer(num_classes=len(label_map), config=self.config)
            if self.n_gpu > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)

            # Initialize training components
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config['training']['lr'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
            scaler = GradScaler(enabled=True)

            # Training loop
            for epoch in range(self.config['training']['epochs']):
                self._train_epoch(model, train_loader, criterion, optimizer, scaler, epoch)
                val_acc = self._validate_epoch(model, valid_loader, criterion, epoch)

                # Update learning rate
                scheduler.step(val_acc)

                # Save checkpoints
                self._save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_acc)

                # Update visualizations
                self._update_visualizations()

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def _train_epoch(self, model, train_loader, criterion, optimizer, scaler, epoch):
        """Run one epoch of training."""
        model.train()
        total_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}") as pbar:
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        self.metrics['train_loss'].append(avg_loss)
        self.logger.info(f"Epoch {epoch+1} - Training Loss: {avg_loss:.4f}")

    def _validate_epoch(self, model, valid_loader, criterion, epoch):
        """Run validation for one epoch."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = total_loss / len(valid_loader)
        val_acc = correct / total

        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)

        self.logger.info(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        return val_acc

    def _save_checkpoint(self, model, optimizer, scheduler, scaler, epoch, val_acc):
        """Save model checkpoint."""
        if val_acc > self.metrics['best_acc']:
            self.metrics['best_acc'] = val_acc
            torch.save(model.state_dict(), self.output_dir / "best_model.pth")

        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'metrics': self.metrics
        }

        torch.save(checkpoint, self.checkpoint_dir / f"checkpoint_{epoch}.pth")

    def _update_visualizations(self):
        """Update training visualizations."""
        plot_training_curves(
            self.metrics['train_loss'],
            self.metrics['val_acc'],
            save_path=self.visualization_dir / "training_curves.png"
        )

def main():
    parser = argparse.ArgumentParser(description="Train audio classification model")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--experiment", help="Experiment name")
    args = parser.parse_args()

    try:
        trainer = TrainingManager(args.config, args.experiment)
        trainer.train()
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
