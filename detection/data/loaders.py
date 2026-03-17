import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset

from detection.data.datasets import IndexDataset


def load_processed(filepath, time_col, lob_columns):
    full = pd.read_parquet(filepath)
    meta_cols = [time_col] + [c for c in lob_columns if c in full.columns]
    meta_set = set(meta_cols)
    feat_cols = [c for c in full.columns if c not in meta_set]
    return full[meta_cols], full[feat_cols]


def scale_and_create_loaders(
    train_feat, val_feat, scaler, model_type, feature_names,
    seq_length, batch_size, target_col, device, fit_scaler=True,
):
    if fit_scaler:
        train_scaled = scaler.fit_transform(train_feat.values.astype(np.float32)).astype(np.float32)
    else:
        train_scaled = scaler.transform(train_feat.values.astype(np.float32)).astype(np.float32)

    val_scaled = scaler.transform(val_feat.values.astype(np.float32)).astype(np.float32)

    train_seqs = create_sequences(train_scaled, seq_length)
    val_seqs = create_sequences(val_scaled, seq_length)

    if len(train_seqs) == 0 or len(val_seqs) == 0:
        return None, None, scaler, feature_names

    target_idx = feature_names.index(target_col)
    train_targets = train_scaled[seq_length:, target_idx][:len(train_seqs)]
    val_targets = val_scaled[seq_length:, target_idx][:len(val_seqs)]

    x_train = torch.tensor(train_seqs, dtype=torch.float32)
    x_val = torch.tensor(val_seqs, dtype=torch.float32)

    if model_type == "pnn":
        y_train = torch.tensor(train_targets, dtype=torch.float32).unsqueeze(1)
        y_val = torch.tensor(val_targets, dtype=torch.float32).unsqueeze(1)
        train_ds = TensorDataset(x_train.reshape(x_train.size(0), -1), y_train)
        val_ds = TensorDataset(x_val.reshape(x_val.size(0), -1), y_val)
    elif model_type == "prae":
        train_ds = IndexDataset(TensorDataset(x_train, x_train))
        val_ds = IndexDataset(TensorDataset(x_val, x_val))
    else:
        train_ds = TensorDataset(x_train, x_train)
        val_ds = TensorDataset(x_val, x_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, scaler, feature_names


def create_sequences(data, seq_length):
    """
    Converts a 2D array (Time, Features) into 3D sequences (Samples, Time, Features).

    Uses a strided view to avoid copying data, then returns a contiguous copy
    only when the input is small enough; otherwise returns the view so that
    downstream code (e.g. DataLoader) can read it without allocating the full
    expanded array up-front.

    Args:
        data: 2D array (Time, Features)
        seq_length: length of sequences for transformer
    Returns:
        sequences: 3D array (Samples, Time, Features)
    """
    n_samples = len(data) - seq_length
    if n_samples <= 0:
        return np.empty((0, seq_length, data.shape[1]), dtype=data.dtype)

    # sliding_window_view gives shape (n_samples+1, Features, seq_length)
    view = sliding_window_view(data, window_shape=seq_length, axis=0)  # (N, F, S)
    view = view[:n_samples].transpose(0, 2, 1)  # (N, S, F)

    # For small arrays return a contiguous copy, for large arrays keep the memory-efficient view.
    nbytes = n_samples * seq_length * data.shape[1] * data.dtype.itemsize
    if nbytes < 2 * (1024 ** 3):  # < 2 GiB -> copy is fine
        return np.ascontiguousarray(view)
    return view

