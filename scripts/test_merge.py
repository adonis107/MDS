"""Merge chunked test outputs and run full post-hoc analysis.

After running test.py with MDS_NUM_CHUNKS > 1, each chunk saves its
scores/preds/meta to test_output/chunk_X/.  This script loads all
chunks, concatenates them, saves the merged results to test_output/,
and runs the full post-hoc analysis (period rates, day summary,
consensus, RCA, proximity).

Usage:
    MDS_YEAR=2015 MDS_NUM_CHUNKS=4 python scripts/test_merge.py
"""
import os
import sys
import glob
import json
import logging
import gc

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from detection.data.loaders import load_processed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_merge")

# ── Configuration (must match test.py) ─────────────────────────
TRAIN_YEAR = os.environ.get("MDS_YEAR", "2015")
NUM_CHUNKS = int(os.environ.get("MDS_NUM_CHUNKS", "4"))
DATA_DIR = os.path.join("data", "processed", "TOTF.PA-book")
RESULTS_DIR = os.path.join("results", str(TRAIN_YEAR))
OUTPUT_DIR = os.path.join("results", str(TRAIN_YEAR), "test_output")

MODEL_TYPES = ["transformer_ocsvm", "pnn", "prae"]
SEQ_LENGTH = 25

LOB_COLUMNS = [
    f"{side}-{typ}-{lvl}"
    for lvl in range(1, 11)
    for side, typ in [("bid", "price"), ("bid", "volume"), ("ask", "price"), ("ask", "volume")]
]

PERIODS = {
    "1st_hour":        (9.0, 10.0),
    "rest_of_morning": (10.0, 12.0),
    "afternoon":       (12.0, 15.5),
    "american_open":   (15.5, 17.5),
}

# ── Build split labels (same logic as test.py) ────────────────
NUM_HOLDOUT = 12
YEAR_FILES = sorted(glob.glob(os.path.join(DATA_DIR, f"{TRAIN_YEAR}*.parquet")))
NUM_TRAIN_DAYS = len(YEAR_FILES) - NUM_HOLDOUT
TEST_PROXIMATE_FILES = YEAR_FILES[NUM_TRAIN_DAYS:NUM_TRAIN_DAYS + 9]
TEST_DISTAL_FILES = YEAR_FILES[NUM_TRAIN_DAYS + 9:]

_proximate_set = set(os.path.basename(f) for f in TEST_PROXIMATE_FILES)
_distal_set = set(os.path.basename(f) for f in TEST_DISTAL_FILES)

# ── Load and merge chunks ─────────────────────────────────────
logger.info("Merging %d chunks from %s", NUM_CHUNKS, OUTPUT_DIR)

all_scores = {mt: [] for mt in MODEL_TYPES}
all_preds = {mt: [] for mt in MODEL_TYPES}
all_period_labels = []
day_names = []
day_split_labels = []
day_boundaries = [0]
processed_test_files = []

for ci in range(NUM_CHUNKS):
    chunk_dir = os.path.join(OUTPUT_DIR, f"chunk_{ci}")
    if not os.path.isdir(chunk_dir):
        logger.error("Chunk directory not found: %s", chunk_dir)
        sys.exit(1)

    for mt in MODEL_TYPES:
        all_scores[mt].append(np.load(os.path.join(chunk_dir, f"{mt}_scores.npy")))
        all_preds[mt].append(np.load(os.path.join(chunk_dir, f"{mt}_preds.npy")))
    all_period_labels.append(np.load(os.path.join(chunk_dir, "period_labels.npy"), allow_pickle=True))

    with open(os.path.join(chunk_dir, "test_meta.json")) as f:
        cmeta = json.load(f)
    day_names.extend(cmeta["day_names"])
    day_split_labels.extend(cmeta["day_split_labels"])

    # Rebase day_boundaries from this chunk onto the global offset
    offset = day_boundaries[-1]
    for b in cmeta["day_boundaries"][1:]:
        day_boundaries.append(offset + b)

    # Reconstruct full paths for RCA streaming
    for dn in cmeta["day_names"]:
        processed_test_files.append(os.path.join(DATA_DIR, dn))

    logger.info("  chunk_%d: %d files, %d samples",
                ci, len(cmeta["day_names"]),
                cmeta["day_boundaries"][-1] - cmeta["day_boundaries"][0]
                if len(cmeta["day_boundaries"]) > 1 else 0)

for mt in MODEL_TYPES:
    all_scores[mt] = np.concatenate(all_scores[mt])
    all_preds[mt] = np.concatenate(all_preds[mt])
period_labels_seq = np.concatenate(all_period_labels)

total_samples = len(next(iter(all_scores.values())))
logger.info("Merged: %d total samples across %d files", total_samples, len(day_names))
for mt in MODEL_TYPES:
    n_anom = all_preds[mt].sum()
    logger.info("  %s: %d anomalies (%.2f%%)", mt, n_anom, 100 * n_anom / total_samples)

# Save merged scores/preds
for mt in MODEL_TYPES:
    np.save(os.path.join(OUTPUT_DIR, f"{mt}_scores.npy"), all_scores[mt])
    np.save(os.path.join(OUTPUT_DIR, f"{mt}_preds.npy"), all_preds[mt])
    logger.info("Saved merged scores & preds for %s", mt)

np.save(os.path.join(OUTPUT_DIR, "period_labels.npy"), period_labels_seq)

meta = {
    "day_names": day_names,
    "day_split_labels": day_split_labels,
    "day_boundaries": day_boundaries,
    "total_samples": total_samples,
    "num_chunks": NUM_CHUNKS,
    "seq_length": SEQ_LENGTH,
}
with open(os.path.join(OUTPUT_DIR, "test_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

# Post-hoc analysis
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

# 7. Root cause analysis (top features per model) - streaming
rca_rows = []

for mt in MODEL_TYPES:
    scores = all_scores[mt]
    preds = all_preds[mt]
    n = len(scores)
    if n == 0:
        continue

    top_idx_global = int(np.argmax(scores))
    threshold_90 = np.percentile(scores, 90)

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
        feat_slice = feat_slice.reindex(columns=feat_col_names, fill_value=0.0)
        feat_arr = feat_slice.values.astype(np.float64)
        del features_day, feat_slice

        day_preds = preds[lo:hi]
        day_scores = scores[lo:hi]

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

        if lo <= top_idx_global < hi:
            top_feat_row = feat_arr[top_idx_global - lo]

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

# 8. Per-split metrics and proximity comparison
split_rows = []
for mt in MODEL_TYPES:
    for split_name in ["test_proximate", "test_distal"]:
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

# 9. Welch t-test: proximate vs. distal
proximity_rows = []
for mt in MODEL_TYPES:
    rates_A = []
    rates_B = []
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
logger.info("All merged results saved to %s", OUTPUT_DIR)
logger.info("Files:")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.isfile(fpath):
        size_kb = os.path.getsize(fpath) / 1024
        logger.info("  %s (%.1f KB)", fname, size_kb)
logger.info("Merge complete.")
