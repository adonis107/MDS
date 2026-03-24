# %% [markdown]
# # Testing of Sequentially Trained Models
#
# Evaluate the three trained model pipelines on held-out test files:
# - **Transformer + OC-SVM**: OC-SVM decision function on latent representations.
# - **PNN + Gain Calc**: Spoofing gain from PNN-predicted skew-normal parameters.
# - **PRAE + RFDR**: Rolling False Discovery Rate on reconstruction error.
#
# For each pipeline we save:
# 1. Raw scores from the initial model (Transformer / PNN / PRAE).
# 2. Binary predictions from the threshold method (OC-SVM / Gain Calc / RFDR).
# 3. Per-period anomaly rates and root-cause feature rankings.

# %%
import os
import sys
import glob
import json
import logging

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from detection.data.loaders import create_sequences, load_processed
from detection.data.preprocessing import get_time_frac, assign_period
from detection.models import hybrid, pnn, prae
from detection.models.transformer import BottleneckTransformer
from detection.spoofing.gain import compute_spoofing_gains_batch
from detection.thresholds.rfdr import RollingFalseDiscoveryRate
from detection.trainers.factory import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("testing")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device: %s", DEVICE)

# %% [markdown]
# ## Configuration
#
# ### File selection
# Change `TEST_FILE_PATTERNS` to test on different files.
# Each entry is a glob pattern relative to `DATA_DIR`.
# Examples:
#   - "2015-02-0*"  → all February 2015 files starting with 0
#   - "2017-*"      → all 2017 files
#   - "2015-02-03*" → a single specific day

# %%
# ── Paths ──────────────────────────────────────────────────────────
DATA_DIR = os.path.join("data", "processed", "TOTF.PA-book")
TRAIN_YEAR = "2015"  # "2015" or "2017" — selects which trained models to load
RESULTS_DIR = os.path.join("results", str(TRAIN_YEAR))
OUTPUT_DIR = os.path.join("results", str(TRAIN_YEAR), "test_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Test file selection ────────────────────────────────────────────
# Option 1: Explicit list of patterns (matched against DATA_DIR)
TEST_FILE_PATTERNS = [
    "2009-*",
    "2010-*",
    "2015-02-03*",  # last 3 of 2015
    "2015-02-04*",
    "2015-02-05*",
    "2017-*",       # all of 2017
]

# Build the test file list from patterns
ALL_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
_test_set = set()
for pattern in TEST_FILE_PATTERNS:
    matches = sorted(glob.glob(os.path.join(DATA_DIR, pattern + ".parquet")))
    if not matches:
        # Try without appending .parquet (in case pattern already includes it)
        matches = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    _test_set.update(matches)
TEST_FILES = sorted(_test_set)

if not TEST_FILES:
    logger.error("No test files found for patterns: %s", TEST_FILE_PATTERNS)
    sys.exit(1)

logger.info("Test files (%d):", len(TEST_FILES))
for f in TEST_FILES:
    logger.info("  %s", os.path.basename(f))

# Model / data parameters (must match training)
MODEL_TYPES = ["transformer_ocsvm", "pnn", "prae"]
SEQ_LENGTH = 25
BATCH_SIZE = 64

LOB_COLUMNS = [
    f"{side}-{typ}-{lvl}"
    for lvl in range(1, 11)
    for side, typ in [("bid", "price"), ("bid", "volume"), ("ask", "price"), ("ask", "volume")]
]

# Threshold parameters
# RFDR (PRAE)
RFDR_WINDOW = 500
RFDR_ALPHA = 0.05

# Spoofing gain (PNN)
SPOOF_Q = 4500
SPOOF_q = 100
SPOOF_DELTA_A = 0.0
SPOOF_DELTA_B = 0.01
SPOOF_FEES = {"maker": 0.0, "taker": 0.05}

# Time-of-day periods for Euronext Paris (hours since midnight, CET)
PERIODS = {
    "1st_hour":        (9.0, 10.0),
    "rest_of_morning": (10.0, 12.0),
    "afternoon":       (12.0, 15.5),
    "american_open":   (15.5, 17.5),
}

# %% [markdown]
# ## Load Models, Scalers, Feature Names

# %%
feature_names_map = {}
for mt in MODEL_TYPES:
    feat_path = os.path.join(RESULTS_DIR, f"{mt}_features.txt")
    if os.path.exists(feat_path):
        with open(feat_path) as f:
            feature_names_map[mt] = [line.strip() for line in f if line.strip()]
    else:
        _, _feat_tmp = load_processed(TEST_FILES[0], "xltime", LOB_COLUMNS)
        feature_names_map[mt] = _feat_tmp.columns.tolist()
        del _feat_tmp

logger.info("Feature names loaded for %d models", len(feature_names_map))
for mt, fnames in feature_names_map.items():
    logger.info("  %s: %d features", mt, len(fnames))

loaded_models = {}
loaded_scalers = {}

for model_type in MODEL_TYPES:
    feat_names = feature_names_map[model_type]
    num_features = len(feat_names)
    weights_path = os.path.join(RESULTS_DIR, f"{model_type}_weights.pth")
    model, ocsvm = load_model(model_type, num_features, weights_path, DEVICE, SEQ_LENGTH)
    loaded_models[model_type] = (model, ocsvm)

    scaler_path = os.path.join(RESULTS_DIR, f"{model_type}_scaler.pkl")
    loaded_scalers[model_type] = joblib.load(scaler_path) if os.path.exists(scaler_path) else MinMaxScaler()
    logger.info("Loaded model & scaler for %s (%d features)", model_type, num_features)

# %% [markdown]
# ## Score Test Files

# %%
# Per-model accumulators
all_scores = {mt: [] for mt in MODEL_TYPES}   # raw model scores
all_preds = {mt: [] for mt in MODEL_TYPES}     # binary predictions from threshold
all_period_labels_seq = []
all_feat_values_seq = []
day_boundaries = [0]
day_names = []

for file_idx, test_file in enumerate(TEST_FILES):
    day_name = os.path.basename(test_file)
    day_names.append(day_name)
    logger.info("=" * 70)
    logger.info("Test file %d/%d: %s", file_idx + 1, len(TEST_FILES), day_name)

    df_day, features_day = load_processed(test_file, "xltime", LOB_COLUMNS)

    time_frac_day = get_time_frac(df_day)[:len(features_day)]
    period_labels_day = assign_period(time_frac_day, PERIODS)

    spread_raw_day = (df_day["ask-price-1"] - df_day["bid-price-1"]).values

    n_seq_day = len(features_day) - SEQ_LENGTH

    period_labels_day_seq = period_labels_day[SEQ_LENGTH: SEQ_LENGTH + n_seq_day]
    all_period_labels_seq.append(period_labels_day_seq)

    feat_values_day_seq = features_day.iloc[SEQ_LENGTH: SEQ_LENGTH + n_seq_day].reset_index(drop=True)
    all_feat_values_seq.append(feat_values_day_seq)

    logger.info("Day rows: %d → %d sequences", len(features_day), n_seq_day)

    for model_type in MODEL_TYPES:
        feat_names = feature_names_map[model_type]
        scaler = loaded_scalers[model_type]
        model, ocsvm = loaded_models[model_type]

        feat_df = features_day.copy()
        for col in feat_names:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        feat_df = feat_df[feat_names]

        scaled = scaler.transform(feat_df.values.astype(np.float32)).astype(np.float32)
        sequences = create_sequences(scaled, SEQ_LENGTH)

        # ── Transformer + OC-SVM ───────────────────────────────────
        if model_type == "transformer_ocsvm":
            x_tensor = torch.tensor(sequences, dtype=torch.float32)
            ds = TensorDataset(x_tensor, x_tensor)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
            if ocsvm is not None:
                detector = hybrid.TransformerOCSVM.__new__(hybrid.TransformerOCSVM)
                detector.transformer = model
                detector.ocsvm = ocsvm
                scores = detector.predict(loader)
            else:
                scores_list = []
                model.eval()
                with torch.no_grad():
                    for batch in loader:
                        x = batch[0].to(DEVICE)
                        rec = model(x)
                        err = torch.mean((x - rec) ** 2, dim=(1, 2)).cpu().numpy()
                        scores_list.append(err)
                scores = np.concatenate(scores_list)
            del x_tensor, ds, loader
            preds = (scores > 0).astype(int)

        # PNN + Spoofing Gain
        elif model_type == "pnn":
            target_col = "log_return"
            all_mu, all_sigma, all_alpha = [], [], []
            n_seqs = len(sequences)

            model.eval()
            with torch.no_grad():
                for start in range(0, n_seqs, BATCH_SIZE):
                    end = min(start + BATCH_SIZE, n_seqs)
                    x_batch = torch.tensor(
                        np.ascontiguousarray(sequences[start:end]),
                        dtype=torch.float32,
                    ).reshape(end - start, -1).to(DEVICE)
                    mu, sigma, alpha = model(x_batch)
                    all_mu.append(mu.cpu().numpy().flatten())
                    all_sigma.append(sigma.cpu().numpy().flatten())
                    all_alpha.append(alpha.cpu().numpy().flatten())

            mu_arr = np.concatenate(all_mu)
            sigma_arr = np.concatenate(all_sigma)
            alpha_arr = np.concatenate(all_alpha)

            spread_seq = spread_raw_day[SEQ_LENGTH: SEQ_LENGTH + len(mu_arr)]
            if len(spread_seq) < len(mu_arr):
                spread_seq = np.pad(spread_seq, (0, len(mu_arr) - len(spread_seq)), mode="edge")
            spread_seq = np.abs(spread_seq)
            spread_seq = np.where(spread_seq > 0, spread_seq, 1e-4)

            scores = compute_spoofing_gains_batch(
                mu_arr, sigma_arr, alpha_arr, spread_seq,
                delta_a=SPOOF_DELTA_A, delta_b=SPOOF_DELTA_B,
                Q=SPOOF_Q, q=SPOOF_q,
                fees=SPOOF_FEES, side="ask",
            )
            preds = (scores > 0).astype(int)

        # PRAE + RFDR
        elif model_type == "prae":
            x_tensor = torch.tensor(sequences, dtype=torch.float32)
            ds = TensorDataset(x_tensor, x_tensor)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
            scores_list = []
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    x = batch[0].to(DEVICE)
                    rec, _ = model(x, training=False)
                    err = torch.sum((x - rec) ** 2, dim=tuple(range(1, x.dim()))).cpu().numpy()
                    scores_list.append(err)
            scores = np.concatenate(scores_list)
            del x_tensor, ds, loader

            rfdr = RollingFalseDiscoveryRate(window_size=RFDR_WINDOW, alpha=RFDR_ALPHA)
            preds = np.zeros(len(scores), dtype=int)
            for i, s in enumerate(scores):
                is_anom, _ = rfdr.process_new_score(float(s))
                preds[i] = int(is_anom)

        all_scores[model_type].append(scores)
        all_preds[model_type].append(preds)

        logger.info("  %s: %d anomalies / %d samples (%.2f%%)",
                     model_type, preds.sum(), len(preds), 100 * preds.mean())

    day_boundaries.append(day_boundaries[-1] + n_seq_day)

# Concatenate across days
for mt in MODEL_TYPES:
    all_scores[mt] = np.concatenate(all_scores[mt])
    all_preds[mt] = np.concatenate(all_preds[mt])

period_labels_seq = np.concatenate(all_period_labels_seq)
feat_values_seq = pd.concat(all_feat_values_seq, ignore_index=True)

total_samples = len(next(iter(all_scores.values())))
logger.info("Scoring complete: %d total samples across %d test files.", total_samples, len(TEST_FILES))
for mt in MODEL_TYPES:
    n_anom = all_preds[mt].sum()
    logger.info("  %s: %d anomalies (%.2f%%)", mt, n_anom, 100 * n_anom / total_samples)

# %% [markdown]
# ## Save Results

# %%
# 1. Per-model raw scores and predictions
for mt in MODEL_TYPES:
    np.save(os.path.join(OUTPUT_DIR, f"{mt}_scores.npy"), all_scores[mt])
    np.save(os.path.join(OUTPUT_DIR, f"{mt}_preds.npy"), all_preds[mt])
    logger.info("Saved scores & preds for %s", mt)

# 2. Period labels
np.save(os.path.join(OUTPUT_DIR, "period_labels.npy"), period_labels_seq)

# 3. Day boundaries and names
meta = {
    "day_names": day_names,
    "day_boundaries": day_boundaries,
    "test_files": [os.path.basename(f) for f in TEST_FILES],
    "total_samples": total_samples,
    "seq_length": SEQ_LENGTH,
    "batch_size": BATCH_SIZE,
    "rfdr_window": RFDR_WINDOW,
    "rfdr_alpha": RFDR_ALPHA,
    "spoof_Q": SPOOF_Q,
    "spoof_q": SPOOF_q,
}
with open(os.path.join(OUTPUT_DIR, "test_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

# 4. Per-period anomaly rates
rows = []
for mt in MODEL_TYPES:
    preds = all_preds[mt]
    n = min(len(preds), len(period_labels_seq))
    for period_name in PERIODS:
        mask = period_labels_seq[:n] == period_name
        total = mask.sum()
        if total == 0:
            continue
        n_anom = preds[:n][mask].sum()
        rows.append({
            "model": mt,
            "period": period_name,
            "total": int(total),
            "anomalies": int(n_anom),
            "rate_pct": round(100 * n_anom / total, 4),
        })

period_df = pd.DataFrame(rows)
period_df.to_csv(os.path.join(OUTPUT_DIR, "anomaly_rates_by_period.csv"), index=False)
logger.info("Saved anomaly rates by period.")

# 5. Per-day summary
day_rows = []
for day_idx, day_name in enumerate(day_names):
    lo = day_boundaries[day_idx]
    hi = day_boundaries[day_idx + 1]
    n_day = hi - lo
    for mt in MODEL_TYPES:
        day_preds = all_preds[mt][lo:hi]
        day_scores = all_scores[mt][lo:hi]
        n_anom = int(day_preds.sum())
        day_rows.append({
            "day": day_name,
            "model": mt,
            "n_samples": n_day,
            "n_anomalies": n_anom,
            "rate_pct": round(100 * n_anom / n_day, 4) if n_day > 0 else 0.0,
            "mean_score": float(day_scores.mean()),
            "max_score": float(day_scores.max()),
        })

day_df = pd.DataFrame(day_rows)
day_df.to_csv(os.path.join(OUTPUT_DIR, "anomaly_rates_by_day.csv"), index=False)
logger.info("Saved anomaly rates by day.")

# 6. Consensus analysis
n_total = min(len(all_preds[mt]) for mt in MODEL_TYPES)
pred_matrix = np.column_stack([all_preds[mt][:n_total] for mt in MODEL_TYPES])
n_models_flagged = pred_matrix.sum(axis=1)

consensus_rows = []
for n_agree in range(len(MODEL_TYPES) + 1):
    count = int((n_models_flagged == n_agree).sum())
    consensus_rows.append({
        "n_models_agreeing": n_agree,
        "sample_count": count,
        "pct": round(100 * count / n_total, 4),
    })

consensus_df = pd.DataFrame(consensus_rows)
consensus_df.to_csv(os.path.join(OUTPUT_DIR, "consensus_agreement.csv"), index=False)
logger.info("Saved consensus agreement.")

# 7. Root cause analysis (top features per model)
rca_rows = []
for mt in MODEL_TYPES:
    scores = all_scores[mt]
    preds = all_preds[mt]
    n = min(len(scores), len(feat_values_seq))

    normal_mask = preds[:n] == 0
    if normal_mask.sum() <= 10:
        continue

    normal_mean = feat_values_seq.iloc[:n][normal_mask].mean()
    normal_std = feat_values_seq.iloc[:n][normal_mask].std().replace(0, 1e-10)

    # Top anomaly
    top_idx = int(np.argmax(scores[:n]))
    top_feat = feat_values_seq.iloc[top_idx]
    z_scores = ((top_feat - normal_mean) / normal_std).abs().sort_values(ascending=False)
    for rank, (feat_name, z) in enumerate(z_scores.head(15).items(), 1):
        rca_rows.append({
            "model": mt,
            "analysis": "top_anomaly",
            "rank": rank,
            "feature": feat_name,
            "z_score": round(float(z), 4),
            "value": round(float(top_feat[feat_name]), 6),
        })

    # Top 10% anomalies mean deviation
    threshold_10pct = np.percentile(scores[:n], 90)
    top10_mask = scores[:n] >= threshold_10pct
    top10_feats = feat_values_seq.iloc[:n][top10_mask]
    top10_mean = top10_feats.mean()
    diff = ((top10_mean - normal_mean) / normal_std).abs().sort_values(ascending=False)
    for rank, (feat_name, d) in enumerate(diff.head(15).items(), 1):
        rca_rows.append({
            "model": mt,
            "analysis": "top10pct_mean",
            "rank": rank,
            "feature": feat_name,
            "z_score": round(float(d), 4),
            "value": round(float(top10_mean[feat_name]), 6),
        })

rca_df = pd.DataFrame(rca_rows)
rca_df.to_csv(os.path.join(OUTPUT_DIR, "root_cause_analysis.csv"), index=False)
logger.info("Saved root cause analysis.")

# Summary
logger.info("=" * 70)
logger.info("All results saved to %s", OUTPUT_DIR)
logger.info("Files:")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    size_kb = os.path.getsize(fpath) / 1024
    logger.info("  %s (%.1f KB)", fname, size_kb)
logger.info("Testing complete.")
