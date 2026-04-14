import numpy as np
from scipy.special import inv_boxcox
from scipy.stats import boxcox as scipy_boxcox
from sklearn.preprocessing import StandardScaler, QuantileTransformer


class EmpiricalBoxCoxScaler:
    """Empirical Box-Cox transformation + z-score normalisation.

    Follows Fabre & Challet (2025): for each feature, estimate the Box-Cox
    lambda via MLE on the training set, apply the transformation, then
    standardise to zero mean and unit variance.

    Non-positive values are handled with a per-feature shift of
    abs(min) + epsilon before transformation.  Extreme values are
    winsorised at the 0.1th and 99.9th training-set percentiles before
    lambda estimation and at transform time.
    """

    def __init__(self, clip_quantiles=(0.001, 0.999), epsilon=1e-6):
        self.clip_quantiles = clip_quantiles
        self.epsilon = epsilon
        # Fitted state
        self.lambdas_ = None
        self.shifts_ = None
        self.clip_lo_ = None
        self.clip_hi_ = None
        self.means_ = None
        self.stds_ = None

    # ------------------------------------------------------------------
    def fit(self, X):
        """Fit the scaler on training data *X* (N x d)."""
        X = np.asarray(X, dtype=np.float64)
        if np.isnan(X).any():
            raise ValueError(
                "Input contains NaN values. Handle missing data before fitting."
            )

        n_features = X.shape[1]
        self.clip_lo_ = np.percentile(X, self.clip_quantiles[0] * 100, axis=0)
        self.clip_hi_ = np.percentile(X, self.clip_quantiles[1] * 100, axis=0)

        X_clipped = np.clip(X, self.clip_lo_, self.clip_hi_)

        # Per-feature shift to ensure strictly positive values
        mins = X_clipped.min(axis=0)
        self.shifts_ = np.where(mins <= 0, np.abs(mins) + self.epsilon, 0.0)

        X_pos = X_clipped + self.shifts_

        self.lambdas_ = np.ones(n_features, dtype=np.float64)
        transformed = np.empty_like(X_pos)

        for j in range(n_features):
            col = X_pos[:, j]
            # Constant feature after clipping: skip transformation
            if col.max() - col.min() < self.epsilon:
                self.lambdas_[j] = 1.0
                transformed[:, j] = col
                continue
            try:
                transformed[:, j], self.lambdas_[j] = scipy_boxcox(col)
            except Exception:
                # Fallback: identity (lambda=1 means (x^1 - 1)/1 = x - 1)
                self.lambdas_[j] = 1.0
                transformed[:, j] = col - 1.0

        self.means_ = transformed.mean(axis=0)
        self.stds_ = transformed.std(axis=0)
        # Avoid division by zero for constant-after-transform features
        self.stds_[self.stds_ < self.epsilon] = 1.0

        return self

    # ------------------------------------------------------------------
    def transform(self, X):
        """Apply the fitted transformation to *X*."""
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        X_clipped = np.clip(X, self.clip_lo_, self.clip_hi_)
        X_pos = X_clipped + self.shifts_

        n_features = X.shape[1]
        transformed = np.empty_like(X_pos)
        for j in range(n_features):
            lam = self.lambdas_[j]
            col = X_pos[:, j]
            if np.abs(lam) < 1e-10:
                transformed[:, j] = np.log(col)
            else:
                transformed[:, j] = (np.power(col, lam) - 1.0) / lam

        return ((transformed - self.means_) / self.stds_).astype(np.float32)

    # ------------------------------------------------------------------
    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    # ------------------------------------------------------------------
    def inverse_transform(self, X_scaled):
        """Reverse the transformation to recover original-scale values."""
        self._check_fitted()
        X_scaled = np.asarray(X_scaled, dtype=np.float64)

        # Undo standardisation
        transformed = X_scaled * self.stds_ + self.means_

        n_features = transformed.shape[1]
        X_pos = np.empty_like(transformed)
        for j in range(n_features):
            lam = self.lambdas_[j]
            X_pos[:, j] = inv_boxcox(transformed[:, j], lam)

        # Undo shift and clip
        return X_pos - self.shifts_

    # ------------------------------------------------------------------
    def _check_fitted(self):
        if self.lambdas_ is None:
            raise ValueError(
                "The scaler has not been fitted yet. Call 'fit' first."
            )


class scaler():
    """
    Quantile transformation + z-score normalization (legacy).
    """
    def __init__(self):
        self.boxcox_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
        self.scaler = StandardScaler()
        self.min_values = None

    def fit(self, X):
        """
        Fit on training data

        Args:
            X: training features (N x d)
        """
        self.min_values = np.min(X, axis=0)
        
        X_positive = X - self.min_values + 1e-6  # Shift to positive for Box-Cox

        self.boxcox_transformer.fit(X_positive)
        X_boxcox = self.boxcox_transformer.transform(X_positive)

        self.scaler.fit(X_boxcox)
        return self
    
    def transform(self, X):
        """
        Transform features

        Args:
            X: features to transform (N x d)
        """
        if self.min_values is None:
            raise ValueError("The preprocessor has not been fitted yet. Call 'fit' with training data first.")
        
        X_positive = X - self.min_values + 1e-6
        X_boxcox = self.boxcox_transformer.transform(X_positive)
        X_scaled = self.scaler.transform(X_boxcox)

        return X_scaled
    
    def fit_transform(self, X):
        """
        Fit and transform features

        Args:
            X: features to fit and transform (N x d)
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_scaled):
        """
        Reverse the transformation to get original values.
        Required for interpreting model outputs.
        """
        if self.min_values is None:
            raise ValueError("The preprocessor has not been fitted yet. Call 'fit' with training data first.")
        
        # Reverse standard scaling
        X_boxcox = self.scaler.inverse_transform(X_scaled)
        # Reverse Box-Cox
        X_positive = self.boxcox_transformer.inverse_transform(X_boxcox)
        # Reverse shift
        X_original = X_positive + self.min_values - 1e-6

        return X_original
