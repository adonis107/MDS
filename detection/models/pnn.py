import numpy as np
import math
import torch
import torch.nn as nn

from detection.base import BaseDeepModel


class SkewedGaussianNLL(nn.Module):
    """
    Negative Log-Likelihood Loss for Skewed Gaussian Distribution
    Based on Equation (20) from Fabre & Challet:
    f(x) = (2/sigma) * phi((x-mu)/sigma) * Phi(alpha * (x-mu)/sigma)
    """
    def __init__(self):
        super(SkewedGaussianNLL, self).__init__()
    
    def _phi(self, z):
        """Standard normal PDF"""
        return (1.0 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * z**2)
    
    def _Phi(self, z):
        """Standard normal CDF"""
        return 0.5 * (1 + torch.erf(z / math.sqrt(2)))

    def forward(self, y_true, mu, sigma, alpha):
        """
        Compute NLL for skewed Gaussian

        Args:
            y_true: Target values (price moves)
            mu: Location parameter
            sigma: Scale parameter (must be > 0)
            alpha: Skewness parameter
        """
        y_true = y_true.view_as(mu)
        z = (y_true - mu) / sigma

        # Skewed Gaussian PDF
        pdf = (2.0 / sigma) * self._phi(z) * self._Phi(alpha * z)
        
        # Negative Log-Likelihood
        log_pdf = -torch.log(pdf + 1e-10)
        return torch.mean(log_pdf)


class PNN(BaseDeepModel):
    """
    PNN: Fabre & Challet
    Architecture: 1 hidden layer, 64 neurons
    Prediction: 3 parameters of the skewed gaussian distribution
    """
    def __init__(self, input_dim, hidden_dim):
        super(PNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 3)  # Output: mu, sigma, alpha
        self.softplus = nn.Softplus()
        self.criterion = SkewedGaussianNLL()

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        """
        Forward pass
        Returns: mu (location), sigma (scale > 0), alpha (skewness)
        """
        h = self.relu(self.fc1(x))
        output = self.fc2(h)

        mu = output[:, 0:1]
        sigma_raw = output[:, 1:2]
        alpha = output[:, 2:3]
        sigma = self.softplus(sigma_raw) + 1e-6
        
        return mu, sigma, alpha
    
    def training_step(self, batch):
        if isinstance(batch, (list, tuple)):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
        else:
            raise ValueError("Batch must be a tuple (x, y)")
        
        mu, sigma, alpha = self.forward(x)
        loss = self.criterion(y, mu, sigma, alpha)
        return loss
    
    def get_anomaly_score(self, batch):
        if isinstance(batch, (list, tuple)):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
        else:
            raise ValueError("Batch must be a tuple (x, y)")
        
        with torch.no_grad():
            mu, sigma, alpha = self.forward(x)
            loss = self.criterion(y, mu, sigma, alpha)
        return loss.cpu().numpy().flatten()

