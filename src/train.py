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
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)

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
        train_dataset = AudioClipDataset(
            data_path=self.config['dataset']['root'],
            split='train',
            config=self.config,
            max_samples=self.config['dataset'].get('max_samples')
        )

        valid_dataset = AudioClipDataset(
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
        """Main training loop with automatic resuming and backup functionality."""
        train_loader, valid_loader, label_map = self.prepare_data()

        # Initialize model
        model = ClipTransformer(num_classes=len(label_map), config=self.config)
        if self.n_gpu > 1:
            model = nn.DataParallel(model)
        model = model.to(self.device)

        # Initialize training components
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=float(self.config['training']['lr'])
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',
            patience=3,
            verbose=True
        )
        scaler = GradScaler(enabled=True)

        # Try to load checkpoint
        start_epoch = 0
        checkpoint = self.checkpoint_manager.load_latest()
        if checkpoint:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            self.metrics = checkpoint['metrics']
            start_epoch = checkpoint['epoch'] + 1
            self.logger.info(f"Resuming training from epoch {start_epoch}")

        # Initialize backup manager
        backup_dir = self.output_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_manager = PeriodicBackup(backup_dir)

        # Training loop
        try:
            for epoch in range(start_epoch, self.config['training']['epochs']):
                self.logger.info(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
                
                # Training phase
                train_loss = self._train_epoch(
                    model=model,
                    train_loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch
                )
                self.metrics['train_loss'].append(train_loss)

                # Validation phase
                val_loss, val_acc = self._validate_epoch(
                    model=model,
                    valid_loader=valid_loader,
                    criterion=criterion,
                    epoch=epoch
                )
                self.metrics['val_loss'].append(val_loss)
                self.metrics['val_acc'].append(val_acc)

                # Update learning rate
                scheduler.step(val_acc)
                current_lr = optimizer.param_groups[0]['lr']
                self.logger.info(f"Current learning rate: {current_lr:.2e}")

                # Check if this is the best model
                is_best = val_acc > self.metrics.get('best_acc', 0)
                if is_best:
                    self.metrics['best_acc'] = val_acc
                    self.logger.info(f"New best validation accuracy: {val_acc:.4f}")

                # Save state
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'metrics': self.metrics,
                    'label_map': label_map,
                    'config': self.config
                }

                # Regular checkpoint save
                self.checkpoint_manager.save(state, epoch, is_best)

                # Periodic backup
                backup_manager.backup(state)

                # Update visualizations
                self._update_visualizations()

                # Early stopping check
                if current_lr < self.config['training'].get('min_lr', 1e-6):
                    self.logger.info("Learning rate below minimum threshold. Stopping training.")
                    break

        except KeyboardInterrupt:
            self.logger.info("\nTraining interrupted by user. Saving checkpoint...")
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'metrics': self.metrics,
                'label_map': label_map,
                'config': self.config
            }
            self.checkpoint_manager.save(state, epoch)
            self.logger.info("Checkpoint saved. Exiting...")
            sys.exit(0)

        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

        finally:
            # Save final model state regardless of how we got here
            self.logger.info("Saving final model state...")
            final_state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'metrics': self.metrics,
                'label_map': label_map,
                'config': self.config
            }
            torch.save(final_state, self.output_dir / "final_model.pth")
            
            # Create final training report
            self._create_training_report()

    def _create_training_report(self):
        """Create a summary report of the training run."""
        report = {
            'experiment_name': self.experiment_name,
            'total_epochs': len(self.metrics['train_loss']),
            'best_validation_accuracy': self.metrics['best_acc'],
            'final_training_loss': self.metrics['train_loss'][-1],
            'final_validation_loss': self.metrics['val_loss'][-1],
            'final_validation_accuracy': self.metrics['val_acc'][-1],
            'training_completed': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save report
        report_path = self.output_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)

        self.logger.info("\nTraining Summary:")
        for key, value in report.items():
            self.logger.info(f"{key}: {value}")


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

        self.logger.info(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        return val_loss, val_acc  # Return both values

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

class CheckpointManager:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.latest_checkpoint = None
        self._find_latest_checkpoint()

    def _find_latest_checkpoint(self) -> None:
        """Find the latest checkpoint in the directory."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pth"))
        if checkpoints:
            # Sort by modification time
            self.latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)

    def save(self, state: dict, epoch: int, is_best: bool = False) -> None:
        """Save checkpoint with automatic cleanup of old checkpoints."""
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{epoch}.pth"
        torch.save(state, checkpoint_path)
        self.latest_checkpoint = checkpoint_path

        # Save best model separately if needed
        if is_best:
            best_path = self.checkpoint_dir.parent / "best_model.pth"
            torch.save(state['model'], best_path)

        # Cleanup old checkpoints - keep only last 3
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pth"), 
                           key=lambda x: x.stat().st_mtime)
        for chk in checkpoints[:-3]:  # Keep last 3 checkpoints
            chk.unlink()

    def load_latest(self) -> Optional[dict]:
        """Load the latest checkpoint if it exists."""
        if self.latest_checkpoint and self.latest_checkpoint.exists():
            return torch.load(self.latest_checkpoint)
        return None

class PeriodicBackup:
    def __init__(self, backup_dir: Path, interval_minutes: int = 30):
        self.backup_dir = backup_dir
        self.interval = interval_minutes * 60  # Convert to seconds
        self.last_backup = time.time()

    def should_backup(self) -> bool:
        return time.time() - self.last_backup >= self.interval

    def backup(self, state: dict) -> None:
        if self.should_backup():
            backup_path = self.backup_dir / f"backup_{int(time.time())}.pth"
            torch.save(state, backup_path)
            self.last_backup = time.time()

            # Cleanup old backups - keep only last 5
            backups = sorted(self.backup_dir.glob("backup_*.pth"), 
                           key=lambda x: x.stat().st_mtime)
            for bak in backups[:-5]:
                bak.unlink()

def main():
    parser = argparse.ArgumentParser(description="Train audio classification model")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--experiment", help="Experiment name")
    args = parser.parse_args()

    trainer = TrainingManager(args.config, args.experiment)
    trainer.train()

if __name__ == "__main__":
    main()
