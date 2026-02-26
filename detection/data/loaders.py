import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


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
    # so we window along axis=0 and then transpose the last two axes.
    view = sliding_window_view(data, window_shape=seq_length, axis=0)  # (N, F, S)
    view = view[:n_samples].transpose(0, 2, 1)  # (N, S, F)

    # For small arrays return a contiguous copy (needed by some downstream ops);
    # for large arrays keep the memory-efficient view.
    nbytes = n_samples * seq_length * data.shape[1] * data.dtype.itemsize
    if nbytes < 2 * (1024 ** 3):  # < 2 GiB → copy is fine
        return np.ascontiguousarray(view)
    return view

