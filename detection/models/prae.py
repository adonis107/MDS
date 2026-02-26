import torch
import torch.nn as nn
import numpy as np

from detection.base import BaseDeepModel
from detection.models.transformer import BottleneckTransformer
from detection.trainers.training import Trainer

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

        if z is not None:
            loss_rec = torch.mean(z * rec_error)
            loss_reg = -self.lambda_reg * torch.mean(z)
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


def grid_search_lambda(train_loader, val_loader, heuristic_lambda, num_train_samples, 
                            num_features, seq_len, device='cuda', epochs=15, learning_rate=1e-3):
    factors = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    candidates = [heuristic_lambda * f for f in factors]

    best_lambda = heuristic_lambda
    best_score = float('inf')

    for candidate in candidates:
        base_ae = BottleneckTransformer(input_dim=num_features, seq_len=seq_len, model_dim=64, num_heads=4, num_layers=2, representation_dim=128)
        model = PRAE(backbone_model=base_ae, num_train_samples=num_train_samples, lambda_reg=candidate).to(device)
        
        # Train
        trainer = Trainer(epochs=epochs, device=device, learning_rate=learning_rate)
        trainer.fit(model, train_loader, val_loader)

        # Evaluate
        model.eval()
        model.to(device)
        total_mse = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                reconstructed, _ = model.backbone(x)
                total_mse += torch.mean((x - reconstructed) ** 2).item()
        
        val_mse = total_mse / len(val_loader)

        if val_mse < best_score:
            best_score = val_mse
            best_lambda = candidate
        
    return best_lambda
