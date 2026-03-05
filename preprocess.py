"""
Preprocess raw LOB CSV files → cleaned parquet with engineered features.

For each raw daily file in data/raw/TOTF.PA-book/:
  1. Fix column-name misalignment (remove 'Unnamed: 1', shift names left,
     drop the trailing all-NaN column).
  2. Clean and filter to market hours  (09:00 - 17:30).
  3. Engineer features (imbalance, dynamics, elasticity, volatility,
     weighted imbalance, event flow, Hawkes, OFI).
  4. Trim warmup rows and align raw LOB with features.
  5. Save as  data/processed/TOTF.PA-book/<date>.parquet
     containing:  xltime  +  raw LOB columns  +  engineered features.
"""

import os
import sys
import glob
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from detection.data.preprocessing import filter_market_hours
from detection.features.dynamics import compute_dynamics, compute_elasticity
from detection.features.event_flow import compute_event_flow
from detection.features.hawkes import compute_hawkes
from detection.features.imbalance import compute_imbalance, compute_weighted_imbalance
from detection.features.ofi import compute_ofi
from detection.features.volatility import compute_volatility

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("preprocess")

# paths
RAW_DIR = os.path.join("data", "raw", "TOTF.PA-book")
OUT_DIR = os.path.join("data", "processed", "TOTF.PA-book")
os.makedirs(OUT_DIR, exist_ok=True)

# constants
TIME_COL = "xltime"
MARKET_OPEN_HOUR = 9.0
MARKET_CLOSE_HOUR = 24.0   # aftermarket for testing
WARMUP_STEPS = 3000
WINDOW = 50

# LOB columns
LOB_COLUMNS = []
for level in range(1, 11):
    LOB_COLUMNS += [
        f"bid-price-{level}",
        f"bid-volume-{level}",
        f"ask-price-{level}",
        f"ask-volume-{level}",
    ]


def fix_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the column-header misalignment in the raw CSV files.

    The raw files have an extra empty header field after 'xltime', which
    pandas reads as 'Unnamed: 1'.  This shifts every subsequent column
    name one position to the right, so the last column is all-NaN.

    Fix: keep 'xltime', rename columns 1..N-2 with the correct LOB names,
    and drop the trailing NaN column.
    """
    new_names = [TIME_COL] + LOB_COLUMNS
    if len(df.columns) != len(new_names) + 1:
        logger.warning(
            "Unexpected column count %d (expected %d). Skipping column fix.",
            len(df.columns), len(new_names) + 1,
        )
        return df

    # Drop the last column (all-NaN artefact) and reassign names
    df = df.iloc[:, :-1].copy()
    df.columns = new_names
    return df


def clean_lob(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by time column."""
    if TIME_COL in df.columns:
        df = df.sort_values(TIME_COL).reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all feature sets from cleaned LOB data.

    Mirrors the engineer_features() used in the training notebook.
    """
    features = pd.DataFrame(index=df.index)

    # base: imbalance
    df = compute_imbalance(df)
    features["L1_Imbalance"] = df["L1_Imbalance"]
    features["L5_Imbalance"] = df["L5_Imbalance"]

    # base: dynamics
    features = compute_dynamics(df, features, window=WINDOW)

    # base: elasticity
    features = compute_elasticity(df, features)

    # base: volatility
    features = compute_volatility(df, features, window=WINDOW)

    # tao: weighted imbalance
    features["Weighted_Imbalance_decreasing"] = compute_weighted_imbalance(
        df, weights=[0.1, 0.1, 0.2, 0.2, 0.4], levels=5)
    features["Weighted_Imbalance_increasing"] = compute_weighted_imbalance(
        df, weights=[0.4, 0.2, 0.2, 0.1, 0.1], levels=5)
    features["Weighted_Imbalance_constant"] = compute_weighted_imbalance(
        df, weights=[0.2, 0.2, 0.2, 0.2, 0.2], levels=5)

    # poutre: event flow / rapidity
    features = compute_event_flow(df, features)

    # hawkes: memory
    features = compute_hawkes(df, features)

    # ofi: order flow imbalance
    features = compute_ofi(df, features)

    # cleanup
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features = features.fillna(0)

    # Warmup trim
    if len(features) > WARMUP_STEPS:
        features = features.iloc[WARMUP_STEPS:].reset_index(drop=True)

    lower = features.quantile(0.001)
    upper = features.quantile(0.999)
    features = features.clip(lower=lower, upper=upper, axis=1)

    # Drop constant columns
    std_devs = features.std()
    drop_cols = std_devs[std_devs < 1e-9].index.tolist()
    if drop_cols:
        features = features.drop(columns=drop_cols)

    return features


def process_file(filepath: str) -> None:
    """Process a single raw daily file and save to parquet."""
    basename = os.path.basename(filepath)
    day_name = basename.replace(".csv.gz", "").replace(".csv", "")
    out_path = os.path.join(OUT_DIR, f"{day_name}.parquet")

    if os.path.exists(out_path):
        logger.info("  ✓ Already processed: %s", out_path)
        return

    logger.info("  Loading %s ...", basename)
    df = pd.read_csv(filepath)

    # Fix column alignment
    df = fix_columns(df)

    # Clean & filter market hours
    df = clean_lob(df)
    df = filter_market_hours(
        df, time_col=TIME_COL,
        market_open_hour=MARKET_OPEN_HOUR,
        market_close_hour=MARKET_CLOSE_HOUR,
    )

    # Engineer features
    features = engineer_features(df)

    # Align raw LOB with features (account for warmup trim)
    if len(df) > len(features):
        df_aligned = df.iloc[-len(features):].reset_index(drop=True)
    else:
        df_aligned = df.reset_index(drop=True)

    # Assemble: xltime + raw LOB + features
    feat_cols = set(features.columns)
    lob_cols = [c for c in df_aligned.columns if c != TIME_COL and c not in feat_cols]
    result = pd.concat(
        [df_aligned[[TIME_COL]], df_aligned[lob_cols], features],
        axis=1,
    )

    # Save
    result.to_parquet(out_path, index=False)
    logger.info("  → Saved %s  (%d rows, %d cols)", out_path, len(result), len(result.columns))


def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv.gz")))
    if not files:
        logger.error("No .csv.gz files found in %s", RAW_DIR)
        sys.exit(1)

    logger.info("Found %d raw files in %s", len(files), RAW_DIR)
    for i, fp in enumerate(files, 1):
        logger.info("[%d/%d] %s", i, len(files), os.path.basename(fp))
        process_file(fp)

    logger.info("Done. Processed files in %s", OUT_DIR)


if __name__ == "__main__":
    main()
