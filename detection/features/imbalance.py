import numpy as np
import pandas as pd


def compute_weighted_imbalance(df, weights=None, levels=5):
    """
    Compute Tao et al. weighted multilevel imbalance from orderbook volumes.
    Spoofing often happens away from the best price (Level 1) to avoid accidental execution.
    Standard L1 imbalance may miss these patterns, so we use a weighted imbalance across multiple levels to capture deeper book dynamics.

    Returns a pandas Series with values clipped to finite and NaNs replaced by 0.
    """
    if weights is None:
        weights = np.array([0.1, 0.1, 0.2, 0.2, 0.4])
    weights = np.asarray(weights)
    if weights.size != levels:
        raise ValueError(f"weights length ({weights.size}) must equal levels ({levels})")

    weighted_bid = sum(weights[i] * df[f"bid-volume-{i+1}"] for i in range(levels))
    weighted_ask = sum(weights[i] * df[f"ask-volume-{i+1}"] for i in range(levels))

    imbalance = weighted_bid / (weighted_bid + weighted_ask)
    # clean numerical issues
    imbalance = imbalance.replace([np.inf, -np.inf], np.nan).fillna(0)
    return imbalance


def compute_imbalance(df):
    ### Imbalances ###
    # Order book imbalance
    # Values close to 1 indicate strong buy pressure, close to -1 indicate sell pressure
    df["L1_Imbalance"] = (df['bid-volume-1'] - df['ask-volume-1']) / (df['bid-volume-1'] + df['ask-volume-1'])
    # Imbalance across top 5 levels, to detect layering (volume deep in the book)
    total_bid_volume_5 = df[[f'bid-volume-{i}' for i in range(1, 6)]].sum(axis=1)
    total_ask_volume_5 = df[[f'ask-volume-{i}' for i in range(1, 6)]].sum(axis=1)
    df["L5_Imbalance"] = (total_bid_volume_5 - total_ask_volume_5) / (total_bid_volume_5 + total_ask_volume_5)

    return df
