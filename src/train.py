"""
Training script for audio classification model.
"""

import os
import sys
import yaml
import logging
import json
import time
import traceback
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

from utils.data import AudioDataset
from utils.model import AudioTransformer
from utils.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    print_classification_report
)

class TrainingManager:
    """Manages the training process for the audio classification model."""

    def __init__(self, config_path: str, experiment_name: Optional[str] = None, 
                 checkpoint_path: Optional[str] = None, fresh_start: bool = False):
        """
        Initialize training manager.

        Args:
            config_path: Path to configuration file
            experiment_name: Optional name for the experiment
            checkpoint_path: Optional path to specific checkpoint to resume from
            fresh_start: If True, start fresh regardless of existing checkpoints
        """
        self.setup_logging()
        self.config = self.load_config(config_path)
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_directories()
        self.setup_device()
        
        self.checkpoint_path = checkpoint_path
        self.fresh_start = fresh_start
        self.metrics = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'best_acc': 0.0}

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
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self._validate_config(config)
        return config

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
        """Prepare training and validation datasets."""
        train_dataset = AudioDataset(
            data_path=self.config['dataset']['root'],
            split='train',
            config=self.config,
            max_samples=self.config['dataset'].get('max_samples')
        )

        valid_dataset = AudioDataset(
            data_path=self.config['dataset']['root'],
            split='dev',
            config=self.config
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=0,
            # num_workers=min(os.cpu_count(), 4),
            # pin_memory=True
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config['training']['batch_size'],
            num_workers=0,
            # num_workers=min(os.cpu_count(), 4),
            # pin_memory=True
        )

        return train_loader, valid_loader, train_dataset.label_map

    def train(self) -> None:
        """Main training loop."""
        try:
            train_loader, valid_loader, label_map = self.prepare_data()
            model = AudioTransformer(num_classes=len(label_map), config=self.config)
            
            if self.n_gpu > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(self.config['training']['lr'])
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=3, verbose=True
            )
            criterion = nn.CrossEntropyLoss()

            # Load checkpoint if available
            start_epoch = 0
            checkpoint = self.load_checkpoint()
            if checkpoint:
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                self.metrics = checkpoint['metrics']
                start_epoch = checkpoint['epoch'] + 1

            # Training loop
            for epoch in range(start_epoch, self.config['training']['epochs']):
                train_loss = self._train_epoch(model, train_loader, criterion, optimizer, epoch)
                val_loss, val_acc = self._validate_epoch(model, valid_loader, criterion, epoch)
                
                # Update metrics
                self.metrics['train_loss'].append(train_loss)
                self.metrics['val_loss'].append(val_loss)
                self.metrics['val_acc'].append(val_acc)
                
                # Check for best model
                is_best = val_acc > self.metrics['best_acc']
                if is_best:
                    self.metrics['best_acc'] = val_acc
                    self.logger.info(f"New best validation accuracy: {val_acc:.4f}")

                # Save checkpoint
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'metrics': self.metrics,
                    'label_map': label_map,
                    'config': self.config
                }
                self.save_checkpoint(state, is_best)

                # Update learning rate
                scheduler.step(val_acc)
                
                # Early stopping check
                if optimizer.param_groups[0]['lr'] < self.config['training'].get('min_lr', 1e-6):
                    self.logger.info("Learning rate below minimum threshold. Stopping training.")
                    break

        except KeyboardInterrupt:
            self.logger.info("\nTraining interrupted by user")
        finally:
            self._create_training_report()

    def _create_training_report(self):
        """Create and log training summary report."""
        report = {
            'experiment_name': self.experiment_name,
            'total_epochs': len(self.metrics['train_loss']),
            'best_validation_accuracy': self.metrics['best_acc'],
            'final_training_loss': self.metrics['train_loss'][-1] if self.metrics['train_loss'] else None,
            'final_validation_loss': self.metrics['val_loss'][-1] if self.metrics['val_loss'] else None,
            'final_validation_accuracy': self.metrics['val_acc'][-1] if self.metrics['val_acc'] else None,
            'training_completed': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        for key, value in report.items():
            self.logger.info(f"{key}: {value}")

    def _train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Run one epoch of training."""
        model.train()
        total_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}") as pbar:
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        self.logger.info(f"Epoch {epoch+1} - Training Loss: {avg_loss:.4f}")

        return avg_loss

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

        self.logger.info(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        return val_loss, val_acc  # Return both values

    def save_checkpoint(self, state: dict, is_best: bool = False) -> None:
        """Save checkpoint with optional best model copy."""
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_latest.pth"
        torch.save(state, checkpoint_path)

        # Save best model separately if needed
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(state, best_path)

    def load_checkpoint(self) -> Optional[dict]:
        """Load checkpoint based on initialization parameters."""
        if self.fresh_start:
            self.logger.info("Starting fresh training run as requested")
            return None

        # Try loading specific checkpoint if provided
        if self.checkpoint_path:
            if os.path.exists(self.checkpoint_path):
                self.logger.info(f"Loading specified checkpoint: {self.checkpoint_path}")
                return torch.load(self.checkpoint_path)
            else:
                raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Try loading latest checkpoint from current experiment
        latest_path = self.checkpoint_dir / "checkpoint_latest.pth"
        if latest_path.exists():
            self.logger.info(f"Resuming from latest checkpoint: {latest_path}")
            return torch.load(latest_path)

        return None

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
    parser.add_argument("--experiment", help="Experiment name (default: timestamp)")
    parser.add_argument("--checkpoint", help="Path to specific checkpoint to resume from")
    parser.add_argument("--fresh", action="store_true", 
                       help="Start training from scratch, ignore existing checkpoints")
    args = parser.parse_args()

    trainer = TrainingManager(
        config_path=args.config,
        experiment_name=args.experiment,
        checkpoint_path=args.checkpoint,
        fresh_start=args.fresh
    )
    trainer.train()

if __name__ == "__main__":
    main()
