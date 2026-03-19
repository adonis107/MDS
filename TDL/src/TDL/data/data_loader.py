import numpy as np
import torch
from torchvision import datasets, transforms


def load_cifar10(n_train: int = 2000, n_test: int = 1000):
    """
    Load and preprocess CIFAR-10 dataset.
    
    Preprocessing:
    - Standardize to zero mean and unit variance
    - No data augmentation
    
    Args:
        n_train: Number of training samples
        n_test: Number of test samples
    
    Returns:
        X_train, Y_train, X_test, Y_test as torch tensors
    """
    # Convert to tensor and normalize
    transform = transforms.Compose([transforms.ToTensor(),])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    
    X_train_full = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    y_train_full = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
    
    X_test_full = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    y_test_full = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])
    
    # Subsample
    train_indices = torch.randperm(len(X_train_full))[:n_train]
    test_indices = torch.randperm(len(X_test_full))[:n_test]
    
    X_train = X_train_full[train_indices]
    y_train = y_train_full[train_indices]
    X_test = X_test_full[test_indices]
    y_test = y_test_full[test_indices]
    
    # Standardize
    mean = X_train.mean(dim=(0, 2, 3), keepdim=True)
    std = X_train.std(dim=(0, 2, 3), keepdim=True)
    
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)
    
    # Convert labels
    # For a label c, its encoding is -0.1*1 + e_c (a vector of -0.1 with 0.9 at index c)
    num_classes = 10
    Y_train = torch.zeros(n_train, num_classes)
    Y_train.fill_(-0.1)
    Y_train.scatter_(1, y_train.unsqueeze(1), 0.9)

    Y_test = torch.zeros(n_test, num_classes)
    Y_test.fill_(-0.1)
    Y_test.scatter_(1, y_test.unsqueeze(1), 0.9)
    
    print(f"Data loaded: Train {X_train.shape}, Test {X_test.shape}")
    print(f"Label range: [{Y_train.min():.4f}, {Y_train.max():.4f}]")
    
    return X_train, Y_train, X_test, Y_test, y_test