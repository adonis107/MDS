"""
Data preprocessing utilities for LOB data.

Handles time-column interpretation, market-hours filtering, and
basic cleaning of raw order-book snapshots.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Euronext Paris default session times (CET / local exchange time)
#   Continuous trading:  09:00 -- 17:30
#   Expressed as fractions of a 24-hour day (Excel *xltime* convention).
# -----------------------------------------------------------------------
_EURONEXT_OPEN_FRAC: float = 9.0 / 24.0      # 0.375
_EURONEXT_CLOSE_FRAC: float = 17.5 / 24.0     # 0.729166...


def filter_market_hours(
    df: pd.DataFrame,
    time_col: str = "xltime",
    market_open_hour: float = 9.0,
    market_close_hour: float = 17.5,
) -> pd.DataFrame:
    """Keep only rows that fall within continuous trading hours.

    The function interprets ``time_col`` as an Excel serial date-time
    (integer part = date, fractional part = time of day) and retains
    rows whose time-of-day fraction lies in
    ``[market_open_hour/24, market_close_hour/24]``.

    Parameters
    ----------
    df : DataFrame
        Raw LOB data containing the time column.
    time_col : str
        Name of the Excel serial-date column (default ``"xltime"``).
    market_open_hour : float
        Market open in hours since midnight (default ``9.0`` for 09:00).
    market_close_hour : float
        Market close in hours since midnight (default ``17.5`` for 17:30).

    Returns
    -------
    DataFrame
        Filtered copy with a reset index.
    """
    if time_col not in df.columns:
        raise KeyError(
            f"Time column '{time_col}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    open_frac = market_open_hour / 24.0
    close_frac = market_close_hour / 24.0

    base_date = np.floor(df[time_col].values)
    time_frac = df[time_col].values - base_date

    mask = (time_frac >= open_frac) & (time_frac <= close_frac)
    n_before = len(df)
    df_filtered = df.loc[mask].reset_index(drop=True)
    n_after = len(df_filtered)

    logger.info(
        "Market-hours filter [%.1f:00 -- %.1f:00]: "
        "kept %d / %d rows (dropped %d pre/post-market).",
        market_open_hour,
        market_close_hour,
        n_after,
        n_before,
        n_before - n_after,
    )
    return df_filtered


def split_first_hour_blocks(
    xltime_vals, features,
    market_open_hour, first_hour_minutes,
    train_block_minutes, val_block_minutes,
):
    time_factor = 1.0 / (24.0 * 60.0)
    base_date = np.floor(xltime_vals[0])
    start_time = base_date + market_open_hour / 24.0

    train_mask = np.zeros(len(features), dtype=bool)
    val_mask = np.zeros(len(features), dtype=bool)
    block_duration = (train_block_minutes + val_block_minutes) * time_factor
    num_blocks = int(first_hour_minutes / (train_block_minutes + val_block_minutes))

    for b in range(num_blocks):
        block_start = start_time + b * block_duration
        train_end = block_start + train_block_minutes * time_factor
        val_end = train_end + val_block_minutes * time_factor
        train_mask |= (xltime_vals >= block_start) & (xltime_vals < train_end)
        val_mask |= (xltime_vals >= train_end) & (xltime_vals < val_end)

    train_features = features.loc[train_mask].reset_index(drop=True)
    val_features = features.loc[val_mask].reset_index(drop=True)
    return train_features, val_features


def get_time_frac(df, time_col="xltime"):
    base_date = np.floor(df[time_col].values)
    return df[time_col].values - base_date


def assign_period(time_frac_arr, periods):
    labels = np.full(len(time_frac_arr), "unknown", dtype=object)
    for name, (open_h, close_h) in periods.items():
        mask = (time_frac_arr >= open_h / 24.0) & (time_frac_arr < close_h / 24.0)
        labels[mask] = name
    return labels


def clean_lob(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic sanity cleaning to a raw LOB DataFrame.

    - Drops fully-NaN artifact columns (e.g. ``Unnamed: 1``).
    - Sorts by the time column if present.

    Parameters
    ----------
    df : DataFrame
        Raw LOB data as loaded from disk.

    Returns
    -------
    DataFrame
        Cleaned copy.
    """
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        logger.info("Dropped artifact columns: %s", unnamed_cols)

    # Sort chronologically
    if "xltime" in df.columns:
        df = df.sort_values("xltime").reset_index(drop=True)

    return df
