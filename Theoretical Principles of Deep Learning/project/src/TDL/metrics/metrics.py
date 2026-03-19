import numpy as np
import torch


def compute_accuracy(Y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    
    Args:
        Y_pred: Predictions [N, num_classes]
        y_true: True labels [N]
    
    Returns:
        Accuracy as percentage
    """
    pred_labels = Y_pred.argmax(dim=1).cpu()
    correct = (pred_labels == y_true).float().mean()
    return correct.item() * 100