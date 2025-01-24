import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from typing import List, Optional
import os

def plot_training_curves(
    train_loss: List[float],
    val_acc: List[float],
    save_path: Optional[str] = None
) -> None:
    """Plot training loss and validation accuracy curves.

    Args:
        train_loss: List of training losses
        val_acc: List of validation accuracies
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 5))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    save_path: Optional[str] = None
) -> None:
    """Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        save_path: Optional path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path)
    plt.close()

def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    output_path: Optional[str] = None
) -> None:
    """Generate and optionally save classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        output_path: Optional path to save the report
    """
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
