import numpy as np
import torch
import torch.nn as nn

class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss does not improve.
    """
    def __init__(self, patience=5, verbose=False, delta=0.0, path='checkpoint.pth'):
        """
        Args:
            patience (int, optional): How long to wait after last time validation loss improved. Defaults to 5.
            verbose (bool, optional): If True, prints a message for each validation loss improvement. Defaults to False.
            delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement. Defaults to 0.0.
            path (str, optional): Path for the checkpoint to be saved to. Defaults to 'checkpoint.pth'.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def reset(self):
        """Reset internal state so the callback can be reused for a new training run."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1

            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.early_stop = False

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

