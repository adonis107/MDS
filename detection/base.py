import torch.nn as nn
from abc import ABC, abstractmethod

class BaseDeepModel(nn.Module):
    @abstractmethod
    def training_step(self, batch):
        """
        Takes a batch, calculates forward pass, returns the loss.
        """
        pass
    
    @abstractmethod
    def get_anomaly_score(self, batch):
        """
        Standardizes output: always returns a 1D array of anomaly scores.
        """
        pass


class BaseDetector(ABC):
    @abstractmethod
    def fit(self, data): pass
    @abstractmethod
    def predict(self, data): pass


class BaseThreshold(ABC):
    @abstractmethod
    def find_threshold(self, scores): pass
    @abstractmethod
    def apply(self, scores): pass