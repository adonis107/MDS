import numpy as np
import torch
import torch.nn as nn
from detection.base import BaseDetector
from detection.models.ocsvm import OCSVM
from detection.models.transformer import BottleneckTransformer

class TransformerOCSVM(BaseDetector):
    def __init__(self, transformer_model, trainer, kernel='rbf', nu=0.01, gamma='auto',
                 n_components=300, sgd_lr=0.01, sgd_epochs=50):
        self.transformer = transformer_model
        self.trainer = trainer
        self.ocsvm = OCSVM(
            kernel=kernel, nu=nu, gamma=gamma,
            n_components=n_components, sgd_lr=sgd_lr, sgd_epochs=sgd_epochs,
        )
        self._latent_bank = []

    @property
    def device(self):
        return self.transformer.device

    @staticmethod
    def _median_heuristic_gamma(X: torch.Tensor, n_subsample: int = 2000) -> float:
        """Compute RBF bandwidth via the median heuristic.

        gamma = 1 / median(â€-x_i - x_jâ€-^2) over a random subsample of pairs.
        This is the standard bandwidth selection rule for kernel methods and
        is robust across different latent-space dimensions / variances.
        """
        n = X.shape[0]
        idx = torch.randperm(n, device=X.device)[:min(n_subsample, n)]
        Z = X[idx]
        dists_sq = torch.cdist(Z, Z, p=2).pow(2)
        mask = torch.triu(torch.ones(len(Z), len(Z), device=X.device, dtype=torch.bool), diagonal=1)
        med = dists_sq[mask].median().item()
        med = max(med, 1e-8)
        return 1.0 / med

    def fit(self, train_loader, val_loader=None):
        print("Training Transformer Autoencoder")
        self.trainer.fit(self.transformer, train_loader, val_loader)

        print("Training OC-SVM (NystrÃ¶m)")
        self.transformer.eval()
        device = self.device
        self.transformer.to(device)
        latent_vectors = []

        with torch.no_grad():
            for batch in train_loader:
                batch = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                latent = self.transformer.get_representation(batch)
                latent_vectors.append(latent)

        new_latent = torch.cat(latent_vectors, dim=0)

        self._latent_bank.append(new_latent)
        X_train_latent = torch.cat(self._latent_bank, dim=0)

        gamma = self._median_heuristic_gamma(X_train_latent)
        self.ocsvm.set_gamma(gamma)

        self.ocsvm.fit(X_train_latent)

    def predict(self, dataloader):
        """Return continuous dissimilarity scores (higher = more anomalous).

        This is the PoutrÃ© et al. (2024) dissimilarity function applied to
        the bottleneck representations extracted by the Transformer encoder.
        """
        self.transformer.eval()
        device = self.device
        self.transformer.to(device)
        latent_vectors = []

        with torch.no_grad():
            for batch in dataloader:
                batch = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                latent = self.transformer.get_representation(batch)
                latent_vectors.append(latent)

        if not latent_vectors:
            return np.empty(0)

        X_test_latent = torch.cat(latent_vectors, dim=0)

        return self.ocsvm.dissimilarity_score(X_test_latent)
    

