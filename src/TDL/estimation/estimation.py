import numpy as np
import torch
import time

from TDL.models.CNTK import CNTKVectorized
from TDL.metrics.metrics import compute_accuracy


def kernel_regression(K_train: torch.Tensor, K_test: torch.Tensor, 
                    Y_train: torch.Tensor, reg: float = 1e-4) -> torch.Tensor:
    """
    Solve kernel regression: Y_pred = K_test @ (K_train + lambda*I)^{-1} @ Y_train
    
    Args:
        K_train: Training kernel matrix [N_train, N_train]
        K_test: Test-Train kernel matrix [N_test, N_train]
        Y_train: Training labels [N_train, num_classes]
        reg: Regularization parameter
    
    Returns:
        Predictions [N_test, num_classes]
    """
    device = K_train.device
    N = K_train.shape[0]
    
    # Add regularization
    K_reg = K_train + reg * torch.eye(N, device=device)
    
    # Use Cholesky decomposition for stability
    try:
        L = torch.linalg.cholesky(K_reg)
        alpha = torch.cholesky_solve(Y_train.to(device), L)
    except:
        alpha = torch.linalg.solve(K_reg, Y_train.to(device))
    
    # Predict
    Y_pred = K_test @ alpha
    
    return Y_pred


def run_experiment(X_train, Y_train, X_test, y_test_labels, 
                  depth: int, use_gap: bool, reg: float = 1e-4):
    """
    Run a single CNTK experiment.
    
    Returns:
        accuracy, computation_time
    """
    print(f"\nRunning: Depth={depth}, GAP={use_gap}")
    
    cntk = CNTKVectorized(depth=depth, use_gap=use_gap)
    
    start_time = time.time()
    
    # Compute training kernel
    print("Computing training kernel...")
    K_train = cntk.compute_kernel(X_train)
    
    # Compute test-train kernel
    print("Computing test kernel...")
    K_test = cntk.compute_kernel(X_test, X_train)
    
    comp_time = time.time() - start_time
    print(f"Kernel computation time: {comp_time:.2f}s")
    
    # Kernel regression
    Y_pred = kernel_regression(K_train, K_test, Y_train, reg=reg)
    
    # Compute accuracy
    accuracy = compute_accuracy(Y_pred, y_test_labels)
    print(f"Accuracy: {accuracy:.2f}%")
    
    return accuracy, comp_time