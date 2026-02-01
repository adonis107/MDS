import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Tuple, Optional


class CNTKVectorized:
    """
    CNTK implementation.
    """
    def __init__(self, depth: int, use_gap: bool = True):
        self.depth = depth
        self.use_gap = use_gap
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    @staticmethod
    def _kappa0(rho: torch.Tensor) -> torch.Tensor:
        """
        Derivative covariance for ReLU: E[sigma'(u) * sigma'(v)].
        """
        rho = torch.clamp(rho, -1.0 + 1e-7, 1.0 - 1e-7)
        return (np.pi - torch.arccos(rho)) / (2 * np.pi)
    

    @staticmethod
    def _kappa1(rho: torch.Tensor) -> torch.Tensor:
        """
        Activation covariance for ReLU (normalized): E[sigma(u) * sigma(v)].
        """
        rho = torch.clamp(rho, -1.0 + 1e-7, 1.0 - 1e-7)
        angle = torch.arccos(rho)
        return (rho * (np.pi - angle) + torch.sqrt(1 - rho**2)) / (2 * np.pi)
    

    def _compute_single_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        """
        Compute kernel value between two images.
        
        Args:
            x1: First image [C, H, W]
            x2: Second image [C, H, W]
        
        Returns:
            Kernel value (scalar)
        """
        C, H, W = x1.shape
        
        # Extract patches using unfold
        unfold = torch.nn.Unfold(kernel_size=3, padding=1)
        patches1 = unfold(x1.unsqueeze(0)).squeeze(0)
        patches2 = unfold(x2.unsqueeze(0)).squeeze(0)

        # Normalize patches
        dim_norm = patches1.shape[0]
        Sigma = (patches1.T @ patches2) / dim_norm

        # Self-variances
        var1 = (patches1 ** 2).sum(dim=0) / dim_norm
        var2 = (patches2 ** 2).sum(dim=0) / dim_norm

        Sigma = Sigma.reshape(H, W, H, W)

        if self.use_gap:
            Theta = torch.zeros_like(Sigma)
        else:
            Theta = Sigma.clone()
        
        pad = 1

        # Layer recursion
        for l in range(self.depth):
            v1_flat = var1.flatten()
            v2_flat = var2.flatten()
            std1 = torch.sqrt(v1_flat + 1e-8) 
            std2 = torch.sqrt(v2_flat + 1e-8) 

            Sigma_flat = Sigma.reshape(H * W, H * W)
            
            # Correlation
            std_outer = std1.unsqueeze(1) * std2.unsqueeze(0)
            rho = Sigma_flat / (std_outer + 1e-8)
            
            # Apply kappa functions
            k0 = self._kappa0(rho)
            k1 = self._kappa1(rho)
            
            # Calculate pointwise values
            Sigma_pointwise = (std_outer * k1 * 2.0).reshape(H, W, H, W)
            Theta_pointwise = (Theta.reshape(H*W, H*W) * k0).reshape(H, W, H, W)
            
            # Diagonal convolution
            Sigma_next = torch.zeros_like(Sigma_pointwise)
            Theta_conv = torch.zeros_like(Theta_pointwise)

            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    # Source slices
                    src_h_start = max(0, -di)
                    src_h_end = H + min(0, -di)
                    src_w_start = max(0, -dj)
                    src_w_end = W + min(0, -dj)

                    # Destination slices
                    dst_h_start = max(0, di)
                    dst_h_end = H + min(0, di)
                    dst_w_start = max(0, dj)
                    dst_w_end = W + min(0, dj)

                    Sigma_next[dst_h_start:dst_h_end, dst_w_start:dst_w_end,
                                dst_h_start:dst_h_end, dst_w_start:dst_w_end] += \
                        Sigma_pointwise[src_h_start:src_h_end, src_w_start:src_w_end,
                                        src_h_start:src_h_end, src_w_start:src_w_end]
                    
                    Theta_conv[dst_h_start:dst_h_end, dst_w_start:dst_w_end,
                                dst_h_start:dst_h_end, dst_w_start:dst_w_end] += \
                        Theta_pointwise[src_h_start:src_h_end, src_w_start:src_w_end,
                                        src_h_start:src_h_end, src_w_start:src_w_end]
                
            # Normalize
            Sigma_next = Sigma_next / 9.0
            Theta_conv = Theta_conv / 9.0

            # Update Theta
            if self.use_gap and l == self.depth - 1:
                Theta = Theta_conv
            else:
                Theta= Sigma_next + Theta_conv
            
            Sigma = Sigma_next
            
            # Update variances
            var1_grid = var1.reshape(1, 1, H, W)
            var2_grid = var2.reshape(1, 1, H, W)

            pool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

            var1 = pool(var1_grid).flatten()
            var2 = pool(var2_grid).flatten()
        
        # Final aggregation
        if self.use_gap:
            # GAP: mean over all patches
            return Theta.mean().item()
        else:
            # Vanilla: sum of diagonal elements
            return torch.diag(Theta.reshape(H * W, H * W)).sum().item()
    
    
    def compute_kernel(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute kernel matrix.
        
        Args:
            X: Images [N1, C, H, W]
            Y: Images [N2, C, H, W] or None for self-kernel
        
        Returns:
            Kernel matrix [N1, N2]
        """
        if Y is None:
            Y = X
            symmetric = True
        else:
            symmetric = False
        
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        N1, N2 = X.shape[0], Y.shape[0]
        K = torch.zeros(N1, N2, device=self.device)
        
        total = N1 * N2 if not symmetric else N1 * (N1 + 1) // 2
        pbar = tqdm(total=total, desc="Computing kernel")
        
        for i in range(N1):
            j_start = i if symmetric else 0
            for j in range(j_start, N2):
                val = self._compute_single_kernel(X[i], Y[j])
                K[i, j] = val
                if symmetric and i != j:
                    K[j, i] = val
                pbar.update(1)
        
        pbar.close()
        return K