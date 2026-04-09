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

import gc
import numpy as np
import pandas as pd
import torch
import joblib
from scipy import stats as scipy_stats
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from detection.data.loaders import create_sequences, load_processed
from detection.data.preprocessing import get_time_frac, assign_period, split_first_hour_blocks
from detection.models import hybrid, pnn, prae
from detection.models.ocsvm import OCSVM
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
TRAIN_YEAR = os.environ.get("MDS_YEAR", "2015")  # override via env for SLURM parallelism
RESULTS_DIR = os.path.join("results", str(TRAIN_YEAR))
OUTPUT_DIR = os.path.join("results", str(TRAIN_YEAR), "test_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Test file selection ────────────────────────────────────────────
# Build test splits from the training year's held-out files and
# out-of-sample years.  The first 9 held-out files form T_A
# (proximate) and the final 3 form T_B (distal).
NUM_HOLDOUT = 12
YEAR_FILES = sorted(glob.glob(os.path.join(DATA_DIR, f"{TRAIN_YEAR}*.parquet")))
NUM_TRAIN_DAYS = len(YEAR_FILES) - NUM_HOLDOUT
TEST_PROXIMATE_FILES = YEAR_FILES[NUM_TRAIN_DAYS:NUM_TRAIN_DAYS + 9]
TEST_DISTAL_FILES = YEAR_FILES[NUM_TRAIN_DAYS + 9:]

# Out-of-sample years (always included)
OOS_PATTERNS = ["2009-*", "2010-*"]
# Cross-year: if training on 2015, also test on 2017 and vice versa
CROSS_YEAR = "2017" if TRAIN_YEAR == "2015" else "2015"
OOS_PATTERNS.append(f"{CROSS_YEAR}-*")

OOS_FILES = []
for pattern in OOS_PATTERNS:
    matches = sorted(glob.glob(os.path.join(DATA_DIR, pattern + ".parquet")))
    if not matches:
        matches = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    OOS_FILES.extend(matches)
OOS_FILES = sorted(set(OOS_FILES))

# Combined test file list with split labels
TEST_FILES = TEST_PROXIMATE_FILES + TEST_DISTAL_FILES + OOS_FILES
TEST_FILES = sorted(set(TEST_FILES))

# Build a mapping: filename -> split label
_proximate_set = set(os.path.basename(f) for f in TEST_PROXIMATE_FILES)
_distal_set = set(os.path.basename(f) for f in TEST_DISTAL_FILES)

def _get_split_label(filepath):
    bn = os.path.basename(filepath)
    if bn in _proximate_set:
        return "test_proximate"
    elif bn in _distal_set:
        return "test_distal"
    else:
        return "out_of_sample"

if not TEST_FILES:
    logger.error("No test files found for TRAIN_YEAR=%s", TRAIN_YEAR)
    sys.exit(1)

logger.info("Test files (%d):", len(TEST_FILES))
logger.info("  T_A (proximate): %d files", len(TEST_PROXIMATE_FILES))
logger.info("  T_B (distal):    %d files", len(TEST_DISTAL_FILES))
logger.info("  Out-of-sample:   %d files", len(OOS_FILES))
for f in TEST_FILES:
    logger.info("  [%s] %s", _get_split_label(f), os.path.basename(f))

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

# Baseline tau for OC-SVM (quantile-based, see Section 3.2.1)
OCSVM_BASELINE_TAU_CONTAMINATION = 0.01

# Spoofing gain (PNN)
SPOOF_Q = 4500
SPOOF_q = 100
SPOOF_DELTA_A = 0.0
SPOOF_DELTA_B = 0.01
# Euronext Paris equity fee schedule for large-cap stocks (TOTF.PA).
# Maker fee ~0 (Euronext offers maker rebates on large caps).
# Taker fee ~0.08% = 0.0008 (typical for large-cap equities on Euronext).
SPOOF_FEES = {"maker": 0.0, "taker": 0.0008}

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

# Compute baseline tau for TF-OC-SVM
# Dissimilarity scores on training data => quantile-based baseline.
# Used as threshold when no EVT/RFDR thresholding has been calibrated.
OCSVM_BASELINE_TAU = 0.0  # fallback if computation fails
_tf_model, _tf_ocsvm = loaded_models.get("transformer_ocsvm", (None, None))
if _tf_model is not None and _tf_ocsvm is not None:
    _tf_scaler = loaded_scalers["transformer_ocsvm"]
    _tf_feat_names = feature_names_map["transformer_ocsvm"]
    _train_files = YEAR_FILES[:NUM_TRAIN_DAYS]
    _train_scores_all = []
    _tf_model.eval()
    logger.info("Computing baseline tau on %d training files (contamination=%.2f%%)...",
                len(_train_files), 100 * OCSVM_BASELINE_TAU_CONTAMINATION)
    with torch.no_grad():
        for _tf in _train_files[:5]:  # first 5 days for speed
            try:
                _df, _feat = load_processed(_tf, "xltime", LOB_COLUMNS)
                for _c in _tf_feat_names:
                    if _c not in _feat.columns:
                        _feat[_c] = 0.0
                _feat = _feat[_tf_feat_names]
                _train_block, _ = split_first_hour_blocks(
                    _df["xltime"].values, _feat, 9.0, 60, 5, 5)
                if len(_train_block) < SEQ_LENGTH + 1:
                    continue
                _scaled = _tf_scaler.transform(_train_block.values.astype(np.float32)).astype(np.float32)
                _seqs = create_sequences(_scaled, SEQ_LENGTH)
                if len(_seqs) == 0:
                    continue
                _x = torch.tensor(_seqs, dtype=torch.float32)
                _ds = TensorDataset(_x, _x)
                _loader = DataLoader(_ds, batch_size=BATCH_SIZE, shuffle=False)
                _det = hybrid.TransformerOCSVM.__new__(hybrid.TransformerOCSVM)
                _det.transformer = _tf_model
                _det.ocsvm = _tf_ocsvm
                _scores_day = _det.predict(_loader)
                _train_scores_all.append(_scores_day)
            except Exception as e:
                logger.warning("Baseline tau: skipping %s: %s", os.path.basename(_tf), e)
    if _train_scores_all:
        _all = np.concatenate(_train_scores_all)
        OCSVM_BASELINE_TAU = OCSVM.fit_baseline_tau(_all, OCSVM_BASELINE_TAU_CONTAMINATION)
        logger.info("Baseline tau = %.6f (%.0f training scores, contamination=%.2f%%)",
                     OCSVM_BASELINE_TAU, len(_all), 100 * OCSVM_BASELINE_TAU_CONTAMINATION)
    else:
        logger.warning("Could not compute baseline tau; using default tau=0.0")
    del _train_scores_all

# %% [markdown]
# ## Score Test Files

# %%
# Per-model accumulators
all_scores = {mt: [] for mt in MODEL_TYPES}   # raw model scores
all_preds = {mt: [] for mt in MODEL_TYPES}     # binary predictions from threshold
all_period_labels_seq = []
day_boundaries = [0]
day_names = []
day_split_labels = []
processed_test_files = []  # track files that were actually scored

for file_idx, test_file in enumerate(TEST_FILES):
    day_name = os.path.basename(test_file)
    split_label = _get_split_label(test_file)
    logger.info("=" * 70)
    logger.info("Test file %d/%d: %s  [%s]", file_idx + 1, len(TEST_FILES), day_name, split_label)

    df_day, features_day = load_processed(test_file, "xltime", LOB_COLUMNS)

    time_frac_day = get_time_frac(df_day)[:len(features_day)]
    period_labels_day = assign_period(time_frac_day, PERIODS)

    spread_raw_day = (df_day["ask-price-1"] - df_day["bid-price-1"]).values

    n_seq_day = len(features_day) - SEQ_LENGTH
    if n_seq_day <= 0:
        logger.warning("Day %s has only %d rows (< SEQ_LENGTH=%d), skipping.",
                       day_name, len(features_day), SEQ_LENGTH)
        del df_day, features_day
        continue

    # Append only for non-skipped files (keeps day_names aligned with day_boundaries)
    day_names.append(day_name)
    day_split_labels.append(split_label)
    processed_test_files.append(test_file)

    period_labels_day_seq = period_labels_day[SEQ_LENGTH: SEQ_LENGTH + n_seq_day]
    all_period_labels_seq.append(period_labels_day_seq)

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
            preds = (scores >= OCSVM_BASELINE_TAU).astype(int)

        # PNN + Spoofing Gain
        elif model_type == "pnn":
            target_col = "log_return"
            all_mu, all_sigma, all_alpha = [], [], []
            n_seqs = len(sequences)

            model.eval()
            with torch.no_grad():
                for start in range(0, n_seqs, BATCH_SIZE):
                    end = min(start + BATCH_SIZE, n_seqs)
                    # PNN uses only the last time step of each sequence
                    # (single-step predictor, matching Fabre & Challet)
                    x_batch = torch.tensor(
                        np.ascontiguousarray(sequences[start:end, -1, :]),
                        dtype=torch.float32,
                    ).to(DEVICE)
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

    # Free per-day memory
    del df_day, features_day, spread_raw_day, period_labels_day, time_frac_day
    gc.collect()

# Concatenate across days
for mt in MODEL_TYPES:
    all_scores[mt] = np.concatenate(all_scores[mt])
    all_preds[mt] = np.concatenate(all_preds[mt])

period_labels_seq = np.concatenate(all_period_labels_seq)
del all_period_labels_seq

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
    "day_split_labels": day_split_labels,
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
    split = day_split_labels[day_idx]
    for mt in MODEL_TYPES:
        day_preds = all_preds[mt][lo:hi]
        day_scores = all_scores[mt][lo:hi]
        n_anom = int(day_preds.sum())
        day_rows.append({
            "day": day_name,
            "split": split,
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

# 7. Root cause analysis (top features per model) — streaming
#    Re-read files one at a time to avoid accumulating all feature data in RAM.
rca_rows = []

for mt in MODEL_TYPES:
    scores = all_scores[mt]
    preds = all_preds[mt]
    n = len(scores)
    if n == 0:
        continue

    top_idx_global = int(np.argmax(scores))
    threshold_90 = np.percentile(scores, 90)

    # Online accumulators (Welford-style sums for mean/std)
    normal_sum = None
    normal_sq_sum = None
    normal_count = 0
    top10_sum = None
    top10_count = 0
    top_feat_row = None
    feat_col_names = None

    logger.info("RCA streaming pass for %s (%d files)...", mt, len(processed_test_files))
    for day_idx, test_file in enumerate(processed_test_files):
        lo = day_boundaries[day_idx]
        hi = day_boundaries[day_idx + 1]
        n_day = hi - lo
        if n_day == 0:
            continue

        _, features_day = load_processed(test_file, "xltime", LOB_COLUMNS)
        feat_slice = features_day.iloc[SEQ_LENGTH: SEQ_LENGTH + n_day]
        if feat_col_names is None:
            feat_col_names = feat_slice.columns.tolist()
        feat_arr = feat_slice.values.astype(np.float64)
        del features_day, feat_slice

        day_preds = preds[lo:hi]
        day_scores = scores[lo:hi]

        # Normal samples (preds == 0)
        nmask = day_preds == 0
        if nmask.any():
            nf = feat_arr[nmask]
            if normal_sum is None:
                normal_sum = nf.sum(axis=0)
                normal_sq_sum = (nf ** 2).sum(axis=0)
            else:
                normal_sum += nf.sum(axis=0)
                normal_sq_sum += (nf ** 2).sum(axis=0)
            normal_count += int(nmask.sum())

        # Top anomaly (single highest-scoring sample globally)
        if lo <= top_idx_global < hi:
            top_feat_row = feat_arr[top_idx_global - lo]

        # Top 10% anomalies
        t10mask = day_scores >= threshold_90
        if t10mask.any():
            t10 = feat_arr[t10mask]
            if top10_sum is None:
                top10_sum = t10.sum(axis=0)
            else:
                top10_sum += t10.sum(axis=0)
            top10_count += int(t10mask.sum())

        del feat_arr
    gc.collect()

    if normal_count <= 10 or feat_col_names is None:
        continue

    normal_mean = normal_sum / normal_count
    normal_var = normal_sq_sum / normal_count - normal_mean ** 2
    normal_std = np.sqrt(np.maximum(normal_var, 0))
    normal_std[normal_std == 0] = 1e-10

    # Top anomaly z-scores
    if top_feat_row is not None:
        z = np.abs((top_feat_row - normal_mean) / normal_std)
        order = np.argsort(z)[::-1]
        for rank, idx in enumerate(order[:15], 1):
            rca_rows.append({
                "model": mt,
                "analysis": "top_anomaly",
                "rank": rank,
                "feature": feat_col_names[idx],
                "z_score": round(float(z[idx]), 4),
                "value": round(float(top_feat_row[idx]), 6),
            })

    # Top 10% mean deviation
    if top10_count > 0 and top10_sum is not None:
        top10_mean = top10_sum / top10_count
        diff = np.abs((top10_mean - normal_mean) / normal_std)
        order = np.argsort(diff)[::-1]
        for rank, idx in enumerate(order[:15], 1):
            rca_rows.append({
                "model": mt,
                "analysis": "top10pct_mean",
                "rank": rank,
                "feature": feat_col_names[idx],
                "z_score": round(float(diff[idx]), 4),
                "value": round(float(top10_mean[idx]), 6),
            })

rca_df = pd.DataFrame(rca_rows)
rca_df.to_csv(os.path.join(OUTPUT_DIR, "root_cause_analysis.csv"), index=False)
logger.info("Saved root cause analysis.")

# %% [markdown]
# ## Proximity Analysis
#
# Compare metrics on $\mathcal{T}_A$ (proximate) vs. $\mathcal{T}_B$ (distal)
# to quantify temporal proximity effects.

# %%
# 8. Per-split metrics and proximity comparison
split_rows = []
for mt in MODEL_TYPES:
    for split_name in ["test_proximate", "test_distal"]:
        # Gather per-day anomaly rates for this split
        split_day_rates = []
        split_day_mean_scores = []
        total_samples_split = 0
        total_anomalies_split = 0
        for day_idx, day_name in enumerate(day_names):
            if day_split_labels[day_idx] != split_name:
                continue
            lo = day_boundaries[day_idx]
            hi = day_boundaries[day_idx + 1]
            n_day = hi - lo
            if n_day == 0:
                continue
            day_preds = all_preds[mt][lo:hi]
            day_scores = all_scores[mt][lo:hi]
            rate = 100.0 * day_preds.sum() / n_day
            split_day_rates.append(rate)
            split_day_mean_scores.append(float(day_scores.mean()))
            total_samples_split += n_day
            total_anomalies_split += int(day_preds.sum())

        if total_samples_split == 0:
            continue

        split_rows.append({
            "model": mt,
            "split": split_name,
            "n_days": len(split_day_rates),
            "n_samples": total_samples_split,
            "n_anomalies": total_anomalies_split,
            "anomaly_rate_pct": round(100.0 * total_anomalies_split / total_samples_split, 4),
            "mean_daily_rate_pct": round(float(np.mean(split_day_rates)), 4),
            "std_daily_rate_pct": round(float(np.std(split_day_rates, ddof=1)), 4) if len(split_day_rates) > 1 else 0.0,
            "mean_daily_score": round(float(np.mean(split_day_mean_scores)), 6),
        })

split_df = pd.DataFrame(split_rows)
split_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_by_split.csv"), index=False)
logger.info("Saved per-split metrics.")

# 9. Welch t-test: proximate vs. distal (per-day anomaly rate)
proximity_rows = []
for mt in MODEL_TYPES:
    rates_A = []  # proximate
    rates_B = []  # distal
    scores_A = []
    scores_B = []
    for day_idx, day_name in enumerate(day_names):
        lo = day_boundaries[day_idx]
        hi = day_boundaries[day_idx + 1]
        n_day = hi - lo
        if n_day == 0:
            continue
        rate = 100.0 * all_preds[mt][lo:hi].sum() / n_day
        mean_s = float(all_scores[mt][lo:hi].mean())
        if day_split_labels[day_idx] == "test_proximate":
            rates_A.append(rate)
            scores_A.append(mean_s)
        elif day_split_labels[day_idx] == "test_distal":
            rates_B.append(rate)
            scores_B.append(mean_s)

    if len(rates_A) >= 2 and len(rates_B) >= 2:
        t_rate, p_rate = scipy_stats.ttest_ind(rates_A, rates_B, equal_var=False)
        t_score, p_score = scipy_stats.ttest_ind(scores_A, scores_B, equal_var=False)
    else:
        t_rate, p_rate = float("nan"), float("nan")
        t_score, p_score = float("nan"), float("nan")

    # Combined (all held-out files from this year, i.e. T_A + T_B)
    combined_rates = rates_A + rates_B
    combined_scores = scores_A + scores_B

    proximity_rows.append({
        "model": mt,
        "metric": "anomaly_rate_pct",
        "proximate_mean": round(float(np.mean(rates_A)), 4) if rates_A else None,
        "distal_mean": round(float(np.mean(rates_B)), 4) if rates_B else None,
        "combined_mean": round(float(np.mean(combined_rates)), 4) if combined_rates else None,
        "welch_t": round(float(t_rate), 4),
        "p_value": round(float(p_rate), 6),
        "n_proximate": len(rates_A),
        "n_distal": len(rates_B),
    })
    proximity_rows.append({
        "model": mt,
        "metric": "mean_score",
        "proximate_mean": round(float(np.mean(scores_A)), 6) if scores_A else None,
        "distal_mean": round(float(np.mean(scores_B)), 6) if scores_B else None,
        "combined_mean": round(float(np.mean(combined_scores)), 6) if combined_scores else None,
        "welch_t": round(float(t_score), 4),
        "p_value": round(float(p_score), 6),
        "n_proximate": len(scores_A),
        "n_distal": len(scores_B),
    })

proximity_df = pd.DataFrame(proximity_rows)
proximity_df.to_csv(os.path.join(OUTPUT_DIR, "proximity_comparison.csv"), index=False)
logger.info("Saved proximity comparison (Welch t-test, T_A vs T_B).")
logger.info("NOTE: small sample sizes (n_A=%d, n_B=%d) limit statistical power.",
            len(TEST_PROXIMATE_FILES), len(TEST_DISTAL_FILES))

# Summary
logger.info("=" * 70)
logger.info("All results saved to %s", OUTPUT_DIR)
logger.info("Files:")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    size_kb = os.path.getsize(fpath) / 1024
    logger.info("  %s (%.1f KB)", fname, size_kb)
logger.info("Testing complete.")
