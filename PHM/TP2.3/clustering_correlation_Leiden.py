from numpy import linalg as LA
import numpy as np
import pandas as pd
from sknetwork.clustering import Leiden
from scipy import sparse


def compute_filtered_C(R):
    # Dimensions
    T, N = R.shape

    # Empirical correlation matrix
    C = np.corrcoef(R, rowvar=False)

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = LA.eig(C)

    # Marcenko-Pastur upper bound
    q = N / T
    lambda_plus = (1 + np.sqrt(q)) ** 2

    # Market mode
    lambda_m = np.max(eigenvalues)

    # Filtered eigenvalues
    mask = (eigenvalues > lambda_plus) & (eigenvalues < lambda_m)
    filtered_eigenvalues = np.where(mask, eigenvalues, 0)

    # Reconstruct the filtered correlation matrix
    C_s = (eigenvectors * filtered_eigenvalues) @ eigenvectors.T

    return C_s


def LeidenCorrelationClustering(C_s):
    # Ensure positivity
    C_s_abs = np.abs(C_s)

    # Remove self-loops
    np.fill_diagonal(C_s_abs, 0)

    # Convert to sparse matrix
    adjacency = sparse.csr_matrix(C_s_abs)

    # Apply Leiden algorithm
    leiden = Leiden()
    labels = leiden.fit_predict(adjacency)

    return labels