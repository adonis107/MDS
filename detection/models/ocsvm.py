"""
One-Class SVM with Nyström RBF kernel approximation - pure PyTorch / CUDA.

Replaces the sklearn ``OneClassSVM`` with a GPU-native pipeline:

1. Nyström approximation - sample m landmark points to build a
    low-rank approximation of the RBF kernel feature map $\phi(x) \in \mathbb{R}^m$.
2. Linear OC-SVM via SGD - a linear separator (w, rho) is trained in
    the approximate feature space on CUDA tensors.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from detection.base import BaseDetector


class OCSVM(nn.Module, BaseDetector):
    """Nyström-approximated One-Class SVM.

    Parameters
    ----------
    nu : float
        Upper bound on the fraction of outliers, in (0, 1].
    gamma : float or ``'auto'``
        RBF bandwidth.  ``'auto'`` -> ``1 / n_features``.
    n_components : int
        Number of Nyström landmarks (approximation rank).
    kernel : str
        Accepted for backward-compatibility; always uses RBF.
    sgd_lr : float
        SGD learning rate for the linear OC-SVM.
    sgd_epochs : int
        Number of SGD training epochs.
    batch_size : int
        Mini-batch size for SGD and batched inference.
    """

    def __init__(
        self,
        nu: float = 0.01,
        gamma="auto",
        n_components: int = 300,
        kernel: str = "rbf",
        sgd_lr: float = 0.01,
        sgd_epochs: int = 50,
        batch_size: int = 256,
    ):
        nn.Module.__init__(self)
        self.nu = nu
        self._gamma = gamma
        self.n_components = n_components
        self.sgd_lr = sgd_lr
        self.sgd_epochs = sgd_epochs
        self.batch_size = batch_size
        self._fitted = False

        # Populated during fit(); registered as buffers for state_dict
        self.register_buffer("_landmarks", None)
        self.register_buffer("_normalization", None)
        self.register_buffer("_w", None)
        self.register_buffer("_rho", None)

    @property
    def gamma(self):
        return self._gamma

    def set_gamma(self, gamma):
        """Set the RBF bandwidth *before* calling :meth:`fit`."""
        self._gamma = gamma

    def set_params(self, **params):
        """sklearn-style parameter setter."""
        for k, v in params.items():
            if k == "gamma":
                self.set_gamma(v)
            else:
                setattr(self, k, v)
        return self

    def _rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """K(X,Y) = exp(-gamma ||x - y||^2),  (n, d) x (m,d) -> (n,m)."""
        X_sq = (X ** 2).sum(dim=1, keepdim=True)
        Y_sq = (Y ** 2).sum(dim=1, keepdim=True)
        dist_sq = X_sq + Y_sq.T - 2.0 * X @ Y.T
        return torch.exp(-self._gamma * dist_sq.clamp(min=0.0))

    def _nystroem_features(self, X: torch.Tensor) -> torch.Tensor:
        """phi(X) via Nyström: K(X, landmarks) @ normalization."""
        K = self._rbf_kernel(X, self._landmarks)
        return K @ self._normalization

    def fit(self, X):
        """Fit the Nyström OC-SVM.

        Parameters
        ----------
        X : numpy array **or** torch Tensor, shape ``(n_samples, n_features)``
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        device = (
            X.device
            if X.is_cuda
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        X = X.to(device)
        n, d = X.shape

        if self._gamma == "auto":
            self._gamma = 1.0 / d

        # landmarks
        m = min(self.n_components, n)
        idx = torch.randperm(n, device=device)[:m]
        landmarks = X[idx].clone()

        # normalization: K_zz = U S U^T ->  norm = U S^{-1/2}
        K_zz = self._rbf_kernel(landmarks, landmarks)
        K_zz += 1e-6 * torch.eye(m, device=device)
        S, U = torch.linalg.eigh(K_zz)
        S = S.clamp(min=1e-8)
        normalization = U @ torch.diag(S.rsqrt())

        self._landmarks = landmarks
        self._normalization = normalization
        self.to(device)

        # transform training data (batched)
        parts = []
        for i in range(0, n, self.batch_size):
            parts.append(self._nystroem_features(X[i : i + self.batch_size]))
        phi = torch.cat(parts, dim=0)

        # linear OC-SVM via SGD
        w = torch.randn(m, device=device) * 0.01
        rho = torch.zeros(1, device=device)
        w.requires_grad_(True)
        rho.requires_grad_(True)

        optim = torch.optim.SGD([w, rho], lr=self.sgd_lr, momentum=0.9)
        ds = TensorDataset(phi.detach())
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.sgd_epochs):
            for (batch,) in loader:
                scores = batch @ w - rho
                hinge = torch.clamp(-scores, min=0.0)
                loss = 0.5 * w.dot(w) - rho + hinge.mean() / self.nu
                optim.zero_grad()
                loss.backward()
                optim.step()

        self._w = w.detach()

        # set rho as the nu-quantile of training projections omega x phi(x).
        # KKT condition: exactly nu fraction of training points lie outside the boundary
        # (decision_function < 0), regardless of SGD convergence quality.
        with torch.no_grad():
            proj_parts = []
            for i in range(0, phi.shape[0], self.batch_size):
                proj_parts.append(phi[i : i + self.batch_size] @ self._w)
            all_proj = torch.cat(proj_parts)
            self._rho = torch.quantile(all_proj, self.nu).unsqueeze(0)

        self._fitted = True

    def decision_function(self, X):
        """Signed distance to the separating hyper-plane.

        Positive → inlier, negative → outlier (sklearn convention).

        Parameters
        ----------
        X : numpy array or torch Tensor, shape ``(n_samples, n_features)``

        Returns
        -------
        numpy 1-D array of length *n_samples*
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self._landmarks.device)

        parts = []
        with torch.no_grad():
            for i in range(0, X.shape[0], self.batch_size):
                phi = self._nystroem_features(X[i : i + self.batch_size])
                parts.append(phi @ self._w - self._rho)
        return torch.cat(parts).cpu().numpy()

    def dissimilarity_score(self, X):
        r"""Continuous dissimilarity score from Poutré et al. (2024), §3.5.

        Defined as the negated OC-SVM decision function:

        .. math::

            \mathrm{dissimilarity}(\mathbf{z}) = \rho
            - \mathbf{w}^\top \tilde{\Phi}(\mathbf{z})
            = -f(\mathbf{z})

        Higher values indicate greater deviation from the learned support
        of normal data. Positive values correspond to points outside the
        decision boundary (outliers); negative values to inliers.

        Range: :math:`\mathbb{R}` (unbounded).

        Reference: Poutré, C., Chételat, D., & Morales, M. (2024).
        *Deep unsupervised anomaly detection in high-frequency markets*.
        J. Finance Data Sci., 10, 100129. Equation in Section 3.5.

        Parameters
        ----------
        X : numpy array or torch Tensor, shape ``(n_samples, n_features)``

        Returns
        -------
        numpy 1-D float array of length *n_samples*
        """
        return -self.decision_function(X)

    def predict(self, X, tau=0.0):
        """Binary anomaly prediction with explicit threshold.

        Parameters
        ----------
        X : numpy array or torch Tensor, shape ``(n_samples, n_features)``
        tau : float
            Detection threshold on the dissimilarity score.
            An observation is flagged anomalous when
            ``dissimilarity_score(X) >= tau``.

        Returns
        -------
        numpy 1-D int array: ``+1`` anomalous, ``-1`` normal.
        """
        scores = self.dissimilarity_score(X)
        return np.where(scores >= tau, 1, -1)

    @staticmethod
    def fit_baseline_tau(scores_train, contamination=0.01):
        """Compute a baseline threshold as a quantile of training scores.

        The baseline tau is the (1 - contamination)-th quantile of the
        training dissimilarity scores.  It serves as a starting threshold
        when no data-driven method (POT, SPOT, DSPOT, RFDR) has been
        calibrated, and as a fallback if those methods fail to fit.

        Parameters
        ----------
        scores_train : array-like, shape ``(n_samples,)``
            Dissimilarity scores computed on the training set.
        contamination : float, default 0.01
            Assumed fraction of anomalies in the training data.

        Returns
        -------
        float
            The baseline threshold.
        """
        scores_train = np.asarray(scores_train, dtype=np.float64)
        return float(np.quantile(scores_train, 1.0 - contamination))
