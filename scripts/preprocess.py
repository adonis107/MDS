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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
MARKET_CLOSE_HOUR = 17.5   # Euronext Paris continuous session ends at 17:30
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
    """Fix the column-header misalignment present in raw files.

    Two variants are observed across the raw files:

    - 42-column files (CSV + some parquets): a spurious second column
      ('Unnamed: 1' / 'V2') is inserted after 'xltime', shifting all LOB
      names one position right and leaving the last column NaN/garbage.
      Fix: drop the last column, then rename all 41.

    - 41-column files (some parquets): the parquet was written with the
      first data row used as column headers (data values as names, e.g.
      '42738.21...', '46.49', …).  No extra column — just wrong names.
      Fix: rename all 41 columns directly.

    If the DataFrame already has the correct column names, it is returned
    unchanged.
    """
    new_names = [TIME_COL] + LOB_COLUMNS

    if list(df.columns) == new_names:
        return df

    if len(df.columns) == len(new_names) + 1:
        df = df.iloc[:, :-1].copy()
        df.columns = new_names
    elif len(df.columns) == len(new_names):
        df = df.copy()
        df.columns = new_names
    else:
        logger.warning(
            "Unexpected column count %d (expected %d or %d). Skipping column fix.",
            len(df.columns), len(new_names), len(new_names) + 1,
        )
    return df


def clean_lob(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by time column and coerce LOB columns to numeric."""
    for col in LOB_COLUMNS:
        if col in df.columns and df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
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
    day_name = basename.replace(".csv.gz", "").replace(".csv", "").replace(".parquet", "")
    out_path = os.path.join(OUT_DIR, f"{day_name}.parquet")

    if os.path.exists(out_path):
        logger.info("Already processed: %s", out_path)
        return

    logger.info("Loading %s ...", basename)
    if filepath.endswith(".parquet"):
        import pyarrow.parquet as pq
        df = pq.ParquetFile(filepath).read().to_pandas()
    else:
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
    logger.info("Saved %s  (%d rows, %d cols)", out_path, len(result), len(result.columns))


def main():
    files = sorted(
        glob.glob(os.path.join(RAW_DIR, "*.csv.gz")) +
        glob.glob(os.path.join(RAW_DIR, "*.parquet"))
    )
    if not files:
        logger.error("No .csv.gz or .parquet files found in %s", RAW_DIR)
        sys.exit(1)

    logger.info("Found %d raw files in %s", len(files), RAW_DIR)
    for i, fp in enumerate(files, 1):
        logger.info("[%d/%d] %s", i, len(files), os.path.basename(fp))
        process_file(fp)

    logger.info("Done. Processed files in %s", OUT_DIR)


if __name__ == "__main__":
    main()
