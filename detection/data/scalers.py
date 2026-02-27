import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer

class scaler():
    """
    Box-Cox transformation + z-score normalization
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
