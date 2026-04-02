import math

import torch
import torch.nn as nn
import numpy as np

from detection.base import BaseDeepModel
from detection.models.transformer import BottleneckTransformer
from detection.trainers.training import Trainer

# Standard normal helpers (used by PRAE analytical regularization)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

def _standard_normal_cdf(x):
    """Φ(x) — CDF of the standard normal distribution."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def _standard_normal_log_pdf(x):
    """log φ(x) — log-PDF of the standard normal distribution."""
    return -0.5 * x * x - _LOG_SQRT_2PI


class PRAE(BaseDeepModel):
    def __init__(self, backbone_model, num_train_samples, lambda_reg=1.0, sigma=0.5):
        super(PRAE, self).__init__()
        self.backbone = backbone_model
        self.lambda_reg = lambda_reg
        self.sigma = sigma
        self.mu = nn.Parameter(torch.full((num_train_samples,), 0.5))

    def forward_backbone(self, x):
        return self.backbone(x)

    def forward(self, x, indices=None, training=True):
        reconstructed = self.backbone(x)
        z = None
        if indices is not None and training:
            mu_batch = self.mu[indices]

            # Stochastic Gate: z[i] = max(0, min(1, mu[i] + epsilon))
            noise = torch.randn_like(mu_batch) * self.sigma
            z = torch.clamp(mu_batch + noise, min=0.0, max=1.0)

        return reconstructed, z

    @staticmethod
    def _expected_z(mu, sigma):
        """Analytical E(z) for a truncated Gaussian gate clamped to [0, 1].

        E(z_i) = σ/√(2π) * (exp(-μ²/(2σ²)) - exp(-(1-μ)²/(2σ²)))
                 + (μ - 1) · Φ((1-μ)/σ)
                 - μ · Φ(-μ/σ)
                 + 1
        """
        a = -mu / sigma          # standardised lower bound
        b = (1.0 - mu) / sigma   # standardised upper bound
        term1 = sigma * (_standard_normal_log_pdf(a).exp()
                         - _standard_normal_log_pdf(b).exp())
        term2 = (mu - 1.0) * _standard_normal_cdf(b)
        term3 = -mu * _standard_normal_cdf(a)
        return term1 + term2 + term3 + 1.0

    def training_step(self, batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                x, indices = batch
            elif len(batch) == 3:
                x, _, indices = batch
            else:
                raise ValueError("Batch must be a tuple of (x, indices) or (x, y, indices)")
        else:
            raise ValueError("Batch must be a tuple of (x, indices) or (x, y, indices)")

        x = x.to(self.device)
        indices = indices.to(self.device)

        is_training = self.training
        reconstructed, z = self.forward(
            x, indices if is_training else None, training=is_training
        )

        rec_error = torch.sum((x - reconstructed) ** 2, dim=tuple(range(1, x.dim())))
        # Clamp rec_error to prevent overflow to Inf.  When mu drifts
        # negative the gate z is clamped to 0, masking that sample out of
        # the loss.  The backbone then receives no gradient to reconstruct
        # it, so its rec_error grows unboundedly.  In IEEE 754,
        # 0.0 * Inf = NaN, which poisons the entire loss and all gradients.
        rec_error = torch.clamp(rec_error, max=1e6)

        if z is not None:
            # PRAE-ℓ₁ loss (Lindenbaum et al.):
            #   L = Σ z_i·‖x_i − x̂_i‖² − λ·Σ E(z_i)
            # Reconstruction term: MC estimate using sampled z
            loss_rec = torch.mean(z * rec_error)
            # Regularization term: analytical E(z) — smooth gradients for all μ
            mu_batch = self.mu[indices]
            ez = self._expected_z(mu_batch, self.sigma)
            loss_reg = -self.lambda_reg * torch.mean(ez)
            loss = loss_rec + loss_reg
        else:
            # Validation: plain reconstruction loss (no stochastic gate)
            loss = torch.mean(rec_error)

        return loss
    
    def get_anomaly_score(self, batch):
        """Compute anomaly scores for unseen data."""
        x = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
        with torch.no_grad():
            reconstructed, _ = self.forward(x, training=False)
            scores = torch.sum((x - reconstructed) ** 2, dim=tuple(range(1, x.dim())))
        return scores

    @property
    def device(self):
        return next(self.backbone.parameters()).device
    

def calculate_heuristic_lambda(train_loader, seq_len, num_features, num_batches=50):
    sq_sum = 0
    count = 0
    for i, batch_data in enumerate(train_loader):
        if i >= num_batches: 
            break
            
        if isinstance(batch_data, (list, tuple)):
            x = batch_data[0]
        else:
            x = batch_data
            
        sq_sum += torch.sum(x**2).item()
        count += x.numel()

    avg_energy = sq_sum / count
    heuristic_lambda = avg_energy * (seq_len * num_features)
    
    return heuristic_lambda


def calculate_reconstruction_lambda(model, train_loader, device='cuda', num_batches=50):
    """Compute lambda from the backbone's current reconstruction error.

    On days 2+ the backbone is already trained, so the raw-energy heuristic
    (ME ≈ Σ‖x‖²/N) vastly overestimates the scale at which rec errors live.
    Instead, we set lambda = mean per-sample reconstruction error, which is the
    natural break-even point: samples with error > lambda have their gates
    pushed toward 0 (anomalous), samples below toward 1 (normal).
    """
    model.eval()
    total_error = 0.0
    n_samples = 0
    with torch.no_grad():
        for i, batch_data in enumerate(train_loader):
            if i >= num_batches:
                break
            x = batch_data[0].to(device) if isinstance(batch_data, (list, tuple)) else batch_data.to(device)
            reconstructed = model.backbone(x)
            rec_error = torch.sum((x - reconstructed) ** 2, dim=tuple(range(1, x.dim())))
            total_error += rec_error.sum().item()
            n_samples += rec_error.shape[0]
    model.train()
    return total_error / n_samples


def grid_search_lambda(train_loader, val_loader, heuristic_lambda, num_train_samples,
                            num_features, seq_len, device='cuda', epochs=15, learning_rate=1e-3,
                            model_dim=128, num_heads=8, num_layers=6, representation_dim=128, dim_feedforward=512):
    """Select lambda_reg by minimising validation reconstruction loss (paper Section 4.2).

    Trains a fresh PRAE for each candidate lambda (multiples of the heuristic)
    and returns the value that yields the lowest reconstruction error on unseen
    validation samples - the tuning scheme proposed in the PRAE literature.
    """
    factors = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    candidates = [heuristic_lambda * f for f in factors]

    best_lambda = heuristic_lambda
    best_score = float('inf')

    for candidate in candidates:
        base_ae = BottleneckTransformer(
            num_features=num_features, sequence_length=seq_len,
            model_dim=model_dim, num_heads=num_heads,
            num_layers=num_layers, representation_dim=representation_dim,
            dim_feedforward=dim_feedforward)
        model = PRAE(backbone_model=base_ae, num_train_samples=num_train_samples,
                     lambda_reg=candidate).to(device)
        
        # Train
        trainer = Trainer(epochs=epochs, device=device, learning_rate=learning_rate)
        trainer.fit(model, train_loader, val_loader)

        # Evaluate reconstruction error on validation set
        model.eval()
        total_mse = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                reconstructed = model.backbone(x)
                total_mse += torch.mean((x - reconstructed) ** 2).item()
        
        val_mse = total_mse / len(val_loader)

        if val_mse < best_score:
            best_score = val_mse
            best_lambda = candidate
        
    return best_lambda
