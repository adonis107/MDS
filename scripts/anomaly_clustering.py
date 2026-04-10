"""
Anomaly Clustering

Cluster anomalies detected by Transformer+OC-SVM, PNN (spoofing gain), and PRAE (RFDR).
Converted from notebook 6. anomaly clustering.ipynb for cluster execution.
"""

import argparse
import os
import glob
import json
import logging
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from torch.utils.data import DataLoader, TensorDataset

from detection.data.loaders import create_sequences, load_processed
from detection.data.preprocessing import get_time_frac, assign_period
from detection.models import hybrid, pnn as pnn_module, prae as prae_module
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
logger = logging.getLogger("anomaly_clustering")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_TYPES = ["transformer_ocsvm", "pnn", "prae"]

LOB_COLUMNS = [
    f"{side}-{typ}-{lvl}"
    for lvl in range(1, 11)
    for side, typ in [
        ("bid", "price"),
        ("bid", "volume"),
        ("ask", "price"),
        ("ask", "volume"),
    ]
]

PERIODS = {
    "1st_hour": (9.0, 10.0),
    "rest_of_morning": (10.0, 12.0),
    "afternoon": (12.0, 15.5),
    "american_open": (15.5, 17.5),
}

SCORE_COLS = ["ocsvm_norm", "pnn_norm", "prae_norm"]


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Anomaly clustering from trained models.")
    p.add_argument("--train-years", nargs="+", type=int, default=[2015, 2017])
    p.add_argument("--data-dir", default=os.path.join("data", "processed", "TOTF.PA-book"))
    p.add_argument("--results-dir", default="results")
    p.add_argument("--seq-length", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    # Spoofing gain
    p.add_argument("--spoof-Q", type=float, default=4500.0)
    p.add_argument("--spoof-q", type=float, default=100.0)
    p.add_argument("--spoof-delta-a", type=float, default=0.0)
    p.add_argument("--spoof-delta-b", type=float, default=0.01)
    p.add_argument("--spoof-maker-fee", type=float, default=0.0)
    p.add_argument("--spoof-taker-fee", type=float, default=0.0008)  # ALIGNED: Euronext Paris fee (report §3.4)
    # RFDR
    p.add_argument("--rfdr-window", type=int, default=500)
    p.add_argument("--rfdr-alpha", type=float, default=0.05)
    # HDBSCAN
    p.add_argument("--hdbscan-min-cluster-frac", type=int, default=20)
    p.add_argument("--hdbscan-max-cluster-size", type=int, default=500)
    p.add_argument("--hdbscan-min-samples", type=int, default=15)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────

def resolve_test_files(data_dir):
    files_2010 = sorted(glob.glob(os.path.join(data_dir, "2010-*.parquet")))
    files_2015 = sorted(glob.glob(os.path.join(data_dir, "2015-*.parquet")))
    files_2017 = sorted(glob.glob(os.path.join(data_dir, "2017-*.parquet")))
    test_files = [
        files_2010[0], files_2010[1], files_2010[2],
        files_2010[-3], files_2010[-2], files_2010[-1],
        files_2015[-3], files_2015[-2], files_2015[-1],
        files_2017[-6], files_2017[-5], files_2017[-4],
        files_2017[-3], files_2017[-2], files_2017[-1],
    ]
    return test_files


def _cache_dir(results_dir, year):
    return os.path.join(results_dir, str(year), "test_output", "cache")


def _cache_path(results_dir, year, basename):
    return os.path.join(_cache_dir(results_dir, year), f"{basename}.npz")


def _is_file_cached(results_dir, year, basename):
    return os.path.exists(_cache_path(results_dir, year, basename))


def _migrate_monolithic_cache(results_dir, year):
    output_dir = os.path.join(results_dir, str(year), "test_output")
    meta_path = os.path.join(output_dir, "test_meta.json")
    if not os.path.exists(meta_path):
        return 0
    with open(meta_path) as f:
        meta = json.load(f)
    day_names = meta.get("day_names", [])
    boundaries = meta.get("day_boundaries", [])
    if len(boundaries) < len(day_names) + 1:
        return 0
    cache_d = _cache_dir(results_dir, year)
    if all(os.path.exists(os.path.join(cache_d, f"{dn}.npz")) for dn in day_names):
        return 0
    try:
        full_scores = {mt: np.load(os.path.join(output_dir, f"{mt}_scores.npy")) for mt in MODEL_TYPES}
        full_preds = {mt: np.load(os.path.join(output_dir, f"{mt}_preds.npy")) for mt in MODEL_TYPES}
    except FileNotFoundError:
        return 0
    os.makedirs(cache_d, exist_ok=True)
    migrated = 0
    for i, dn in enumerate(day_names):
        npz_path = os.path.join(cache_d, f"{dn}.npz")
        if os.path.exists(npz_path):
            continue
        lo, hi = boundaries[i], boundaries[i + 1]
        data = {}
        for mt in MODEL_TYPES:
            data[f"{mt}_scores"] = full_scores[mt][lo:hi]
            data[f"{mt}_preds"] = full_preds[mt][lo:hi]
        np.savez(npz_path, **data)
        migrated += 1
    return migrated


def compute_file_scores(features_day, spread_raw, mid_price_raw, feature_names_map, loaded_models,
                        loaded_scalers, args):
    results = {}
    spoof_fees = {"maker": args.spoof_maker_fee, "taker": args.spoof_taker_fee}

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
        sequences = create_sequences(scaled, args.seq_length)

        if model_type == "transformer_ocsvm":
            x_tensor = torch.tensor(sequences, dtype=torch.float32)
            loader = DataLoader(
                TensorDataset(x_tensor, x_tensor), batch_size=args.batch_size, shuffle=False
            )
            if ocsvm is not None:
                detector = hybrid.TransformerOCSVM.__new__(hybrid.TransformerOCSVM)
                detector.transformer = model
                detector.ocsvm = ocsvm
                scores = detector.predict(loader)
            else:
                scores_list = []
                with torch.no_grad():
                    for batch in loader:
                        x = batch[0].to(DEVICE)
                        rec = model(x)
                        scores_list.append(
                            torch.mean((x - rec) ** 2, dim=(1, 2)).cpu().numpy()
                        )
                scores = np.concatenate(scores_list)
            # Use native OC-SVM decision boundary (dissimilarity >= 0) for broad
            # anomaly coverage in clustering. test.py uses a stricter calibrated
            # baseline tau computed from training data.
            preds = (scores > 0).astype(int)

        elif model_type == "pnn":
            all_mu, all_sigma, all_alpha = [], [], []
            with torch.no_grad():
                for start in range(0, len(sequences), args.batch_size):
                    end = min(start + args.batch_size, len(sequences))
                    x_b = torch.tensor(
                        np.ascontiguousarray(sequences[start:end, -1, :]),
                        dtype=torch.float32,
                    ).to(DEVICE)
                    mu, sigma, alpha = model(x_b)
                    all_mu.append(mu.cpu().numpy().flatten())
                    all_sigma.append(sigma.cpu().numpy().flatten())
                    all_alpha.append(alpha.cpu().numpy().flatten())

            mu_arr = np.concatenate(all_mu)
            sigma_arr = np.concatenate(all_sigma)
            alpha_arr = np.concatenate(all_alpha)

            spread_seq = spread_raw[args.seq_length: args.seq_length + len(mu_arr)]
            mid_seq = mid_price_raw[args.seq_length: args.seq_length + len(mu_arr)]
            if len(spread_seq) < len(mu_arr):
                spread_seq = np.pad(spread_seq, (0, len(mu_arr) - len(spread_seq)), mode="edge")
                mid_seq = np.pad(mid_seq, (0, len(mu_arr) - len(mid_seq)), mode="edge")
            spread_seq = np.where(np.abs(spread_seq) > 0, np.abs(spread_seq), 1e-4)

            # PNN outputs are in log-return units; the spoofing-gain formula
            # (Fabre & Challet) expects EUR price changes.  Convert via the
            # first-order approximation  Δp ≈ r · p_mid.
            mu_eur = mu_arr * mid_seq
            sigma_eur = sigma_arr * mid_seq

            scores = compute_spoofing_gains_batch(
                mu_eur, sigma_eur, alpha_arr, spread_seq,
                delta_a=args.spoof_delta_a, delta_b=args.spoof_delta_b,
                Q=args.spoof_Q, q=args.spoof_q,
                fees=spoof_fees, side="ask",
            )
            preds = (scores > 0).astype(int)

        else:  # prae
            x_tensor = torch.tensor(sequences, dtype=torch.float32)
            loader = DataLoader(
                TensorDataset(x_tensor, x_tensor), batch_size=args.batch_size, shuffle=False
            )
            scores_list = []
            with torch.no_grad():
                for batch in loader:
                    x = batch[0].to(DEVICE)
                    rec, _ = model(x, training=False)
                    scores_list.append(
                        torch.sum((x - rec) ** 2, dim=tuple(range(1, x.dim())))
                        .cpu().numpy()
                    )
            scores = np.concatenate(scores_list)

            rfdr = RollingFalseDiscoveryRate(window_size=args.rfdr_window, alpha=args.rfdr_alpha)
            preds = np.zeros(len(scores), dtype=int)
            for i, s in enumerate(scores):
                is_anom, _ = rfdr.process_new_score(float(s))
                preds[i] = int(is_anom)

        results[f"{model_type}_scores"] = scores
        results[f"{model_type}_preds"] = preds

    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

def save_cluster_pca_plot(train_years, score_data, cluster_data, anom_data, output_dir, random_state):
    CLUSTER_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(2, len(train_years), figsize=(7 * len(train_years), 10))
    if len(train_years) == 1:
        axes = axes.reshape(-1, 1)

    for col, year in enumerate(train_years):
        X_clust = score_data[year]["X_clust"]
        cl = cluster_data[year]
        preds_anom = anom_data[year]["preds_anom"]

        pca = PCA(n_components=2, random_state=random_state)
        X_pca = pca.fit_transform(X_clust)

        ax = axes[0, col]
        for c in cl["unique_clusters"]:
            mask = cl["cluster_labels"] == c
            if c == -1:
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=6, alpha=0.25, color="lightgrey", label="Noise")
            else:
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=10, alpha=0.5,
                           color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)], label=cl["cluster_names"][c])
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title(f"{year}: HDBSCAN ({cl['n_hdb_clusters']} clusters, min_size={cl['min_cluster_size']})")
        ax.legend(fontsize=6, markerscale=2)

        ax = axes[1, col]
        pnn_mask = preds_anom["pnn"].astype(bool)
        non_pnn = ~pnn_mask
        ax.scatter(X_pca[non_pnn, 0], X_pca[non_pnn, 1], s=6, alpha=0.2, color="lightgrey", label="Other anomalies")
        ax.scatter(X_pca[pnn_mask, 0], X_pca[pnn_mask, 1], s=14, alpha=0.7, color="#c0392b",
                   edgecolors="black", linewidths=0.3, label=f"PNN spoofing (n={pnn_mask.sum()})")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title(f"{year}: PNN + Spoofing Gain")
        ax.legend(fontsize=7, markerscale=2)

    plt.tight_layout()
    path = os.path.join(output_dir, "cluster_pca.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def save_detection_rate_heatmap(train_years, score_data, cluster_data, anom_data, output_dir):
    max_clusters = max(len(cluster_data[y]["unique_clusters"]) for y in train_years)
    fig, axes_arr = plt.subplots(1, len(train_years), figsize=(6 * len(train_years), 0.6 * max_clusters + 2))
    if len(train_years) == 1:
        axes_arr = [axes_arr]

    for col, year in enumerate(train_years):
        cl = cluster_data[year]
        score_matrix_norm = score_data[year]["score_matrix_norm"]
        preds_anom = anom_data[year]["preds_anom"]
        n_anom = anom_data[year]["n_anom"]

        profile_rows = []
        for c in cl["unique_clusters"]:
            mask = cl["cluster_labels"] == c
            row = {"Cluster": cl["cluster_names"][c], "N": int(mask.sum()),
                   "Share (%)": round(100 * mask.sum() / n_anom, 2)}
            for j, mt in enumerate(MODEL_TYPES):
                short = mt.replace("transformer_ocsvm", "ocsvm")
                row[f"{short}_score_mean"] = round(float(score_matrix_norm[mask, j].mean()), 4)
                row[f"{short}_det_rate"] = round(float(preds_anom[mt][mask].mean()), 4)
            profile_rows.append(row)

        profile_df = pd.DataFrame(profile_rows).set_index("Cluster")
        logger.info("Year %d profile:\n%s", year, profile_df.to_string())

        det_cols = [c for c in profile_df.columns if c.endswith("_det_rate")]
        det_df = profile_df[det_cols].rename(columns={c: c.replace("_det_rate", "") for c in det_cols})
        ax = axes_arr[col]
        sns.heatmap(det_df.astype(float), annot=True, fmt=".2f", cmap="YlOrRd",
                    vmin=0, vmax=1, ax=ax, linewidths=0.5)
        ax.set_title(f"{year}: Detection rate per model")
        ax.set_xlabel("Model")
        ax.set_ylabel("")

    plt.tight_layout()
    path = os.path.join(output_dir, "detection_rate_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logger.info("Device: %s", DEVICE)
    logger.info("Train years: %s", args.train_years)

    test_files = resolve_test_files(args.data_dir)
    logger.info("Test files: %d", len(test_files))

    output_dir = os.path.join(args.results_dir, "anomaly_clustering")
    os.makedirs(output_dir, exist_ok=True)

    # ── Migrate monolithic caches ────────────────────────────────────────
    for year in args.train_years:
        n_mig = _migrate_monolithic_cache(args.results_dir, year)
        if n_mig > 0:
            logger.info("Migrated %d files from monolithic cache for year %d.", n_mig, year)

    # ── Report cache status ──────────────────────────────────────────────
    for year in args.train_years:
        bns = [os.path.basename(f) for f in test_files]
        cached = [bn for bn in bns if _is_file_cached(args.results_dir, year, bn)]
        missing = [bn for bn in bns if not _is_file_cached(args.results_dir, year, bn)]
        logger.info("Year %d: %d/%d cached, %d to compute", year, len(cached), len(bns), len(missing))

    # ── Load feature names & models ──────────────────────────────────────
    feature_names_by_year = {}
    loaded_models_by_year = {}
    loaded_scalers_by_year = {}

    for year in args.train_years:
        rd = os.path.join(args.results_dir, str(year))
        feat_map = {}
        for mt in MODEL_TYPES:
            feat_path = os.path.join(rd, f"{mt}_features.txt")
            if os.path.exists(feat_path):
                with open(feat_path) as fh:
                    feat_map[mt] = [ln.strip() for ln in fh if ln.strip()]
            else:
                _, tmp = load_processed(test_files[0], "xltime", LOB_COLUMNS)
                feat_map[mt] = tmp.columns.tolist()
        feature_names_by_year[year] = feat_map

        bns = [os.path.basename(f) for f in test_files]
        needs_compute = any(not _is_file_cached(args.results_dir, year, bn) for bn in bns)
        if needs_compute:
            models, scalers = {}, {}
            for mt in MODEL_TYPES:
                weights_path = os.path.join(rd, f"{mt}_weights.pth")
                model, ocsvm = load_model(mt, len(feat_map[mt]), weights_path, DEVICE, args.seq_length)
                models[mt] = (model, ocsvm)
                scaler_path = os.path.join(rd, f"{mt}_scaler.pkl")
                scalers[mt] = joblib.load(scaler_path) if os.path.exists(scaler_path) else MinMaxScaler()
            loaded_models_by_year[year] = models
            loaded_scalers_by_year[year] = scalers
            logger.info("Year %d: models loaded.", year)
        else:
            logger.info("Year %d: fully cached, skipping model loading.", year)

    # ── Score computation / cache loading ────────────────────────────────
    all_scores_by_year = {y: {mt: [] for mt in MODEL_TYPES} for y in args.train_years}
    all_preds_by_year = {y: {mt: [] for mt in MODEL_TYPES} for y in args.train_years}
    all_period_labels = []
    all_feat_values = []
    day_boundaries = [0]

    for test_file in test_files:
        basename = os.path.basename(test_file)
        df_day, features_day = load_processed(test_file, "xltime", LOB_COLUMNS)
        n_seq = len(features_day) - args.seq_length

        if n_seq <= 0:
            logger.warning("File %s too small (%d rows). Skipping.", basename, len(features_day))
            continue

        time_frac_day = get_time_frac(df_day)[: len(features_day)]
        period_labels = assign_period(time_frac_day, PERIODS)
        spread_raw = (df_day["ask-price-1"] - df_day["bid-price-1"]).values
        mid_price_raw = 0.5 * (df_day["ask-price-1"] + df_day["bid-price-1"]).values

        all_period_labels.append(period_labels[args.seq_length: args.seq_length + n_seq])
        all_feat_values.append(
            features_day.iloc[args.seq_length: args.seq_length + n_seq].reset_index(drop=True)
        )
        day_boundaries.append(day_boundaries[-1] + n_seq)

        for year in args.train_years:
            if _is_file_cached(args.results_dir, year, basename):
                cached = np.load(_cache_path(args.results_dir, year, basename))
                for mt in MODEL_TYPES:
                    all_scores_by_year[year][mt].append(cached[f"{mt}_scores"])
                    all_preds_by_year[year][mt].append(cached[f"{mt}_preds"])
                logger.info("[%d] Loaded cached: %s", year, basename)
            else:
                logger.info("[%d] Computing: %s", year, basename)
                res = compute_file_scores(
                    features_day, spread_raw, mid_price_raw,
                    feature_names_by_year[year],
                    loaded_models_by_year[year],
                    loaded_scalers_by_year[year],
                    args,
                )
                for mt in MODEL_TYPES:
                    all_scores_by_year[year][mt].append(res[f"{mt}_scores"])
                    all_preds_by_year[year][mt].append(res[f"{mt}_preds"])
                os.makedirs(_cache_dir(args.results_dir, year), exist_ok=True)
                np.savez(_cache_path(args.results_dir, year, basename), **res)
                logger.info("[%d] Cached: %s", year, basename)

    for year in args.train_years:
        for mt in MODEL_TYPES:
            all_scores_by_year[year][mt] = np.concatenate(all_scores_by_year[year][mt])
            all_preds_by_year[year][mt] = np.concatenate(all_preds_by_year[year][mt])

    period_labels_seq = np.concatenate(all_period_labels)
    feat_values_seq = pd.concat(all_feat_values, ignore_index=True)
    n_total = len(period_labels_seq)
    logger.info("Total: %d sequences from %d files.", n_total, len(test_files))

    # ── Anomaly selection ────────────────────────────────────────────────
    anom_data = {}
    for year in args.train_years:
        pred_matrix = np.column_stack([all_preds_by_year[year][mt][:n_total] for mt in MODEL_TYPES])
        indices = np.where(pred_matrix.sum(axis=1) >= 1)[0]
        anom_data[year] = {
            "anom_indices": indices,
            "n_anom": len(indices),
            "score_matrix_raw": np.column_stack([all_scores_by_year[year][mt][indices] for mt in MODEL_TYPES]),
            "feat_anom": feat_values_seq.iloc[indices].reset_index(drop=True),
            "period_anom": period_labels_seq[indices],
            "preds_anom": {mt: all_preds_by_year[year][mt][indices] for mt in MODEL_TYPES},
        }
        n_anom = len(indices)
        logger.info("Year %d: %d anomalies / %d (%.2f%%)", year, n_anom, n_total, 100 * n_anom / n_total)
        for mt in MODEL_TYPES:
            n_mt = int(all_preds_by_year[year][mt][:n_total].sum())
            logger.info("  %25s: %6d (%.2f%%)", mt, n_mt, 100 * n_mt / n_total)

    # ── Normalise scores ─────────────────────────────────────────────────
    score_data = {}
    for year in args.train_years:
        score_matrix_full_norm = np.zeros((n_total, len(MODEL_TYPES)), dtype=np.float32)
        for j, mt in enumerate(MODEL_TYPES):
            sc = MinMaxScaler()
            score_matrix_full_norm[:, j] = sc.fit_transform(
                all_scores_by_year[year][mt][:n_total].reshape(-1, 1)
            ).flatten()
        indices = anom_data[year]["anom_indices"]
        score_matrix_norm = score_matrix_full_norm[indices]
        score_df = pd.DataFrame(score_matrix_norm, columns=SCORE_COLS)
        score_data[year] = {
            "score_matrix_norm": score_matrix_norm,
            "score_df": score_df,
            "X_clust": score_df.values,
        }
        logger.info("Year %d score distribution:\n%s", year, score_df.describe().round(4).to_string())

    # ── HDBSCAN clustering ───────────────────────────────────────────────
    cluster_data = {}
    for year in args.train_years:
        X_clust = score_data[year]["X_clust"]
        n_anom = anom_data[year]["n_anom"]
        preds_anom = anom_data[year]["preds_anom"]

        min_cluster_size = max(5, min(args.hdbscan_max_cluster_size, n_anom // args.hdbscan_min_cluster_frac))
        hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=args.hdbscan_min_samples)
        cluster_labels = hdb.fit_predict(X_clust)

        unique_clusters = sorted(set(cluster_labels))
        n_hdb_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        n_noise = int((cluster_labels == -1).sum())

        mask_valid = cluster_labels != -1
        if mask_valid.sum() >= 2 and n_hdb_clusters >= 2:
            sil = silhouette_score(X_clust[mask_valid], cluster_labels[mask_valid])
            db = davies_bouldin_score(X_clust[mask_valid], cluster_labels[mask_valid])
        else:
            sil, db = np.nan, np.nan

        pnn_rates = {}
        cluster_names = {}
        for c in unique_clusters:
            mask = cluster_labels == c
            pnn_rates[c] = preds_anom["pnn"][mask].mean() if mask.sum() > 0 else 0.0
            if c == -1:
                cluster_names[c] = f"Noise (n={mask.sum()})"
            else:
                cluster_names[c] = f"C{c} (PNN rate={pnn_rates[c]:.1%})"

        cluster_data[year] = {
            "cluster_labels": cluster_labels,
            "unique_clusters": unique_clusters,
            "n_hdb_clusters": n_hdb_clusters,
            "n_noise": n_noise,
            "cluster_names": cluster_names,
            "min_cluster_size": min_cluster_size,
            "sil": sil,
            "db": db,
        }
        logger.info("Year %d: %d clusters, %d noise (%.1f%%), sil=%.4f, db=%.4f",
                     year, n_hdb_clusters, n_noise, 100 * n_noise / n_anom, sil, db)

    # ── Summary table ────────────────────────────────────────────────────
    for year in args.train_years:
        cl = cluster_data[year]
        preds_anom = anom_data[year]["preds_anom"]
        feat_anom = anom_data[year]["feat_anom"]
        period_anom = anom_data[year]["period_anom"]
        n_anom = anom_data[year]["n_anom"]

        global_mean = feat_anom.mean()
        global_std = feat_anom.std().replace(0, 1e-10)

        summary_rows = []
        for c in cl["unique_clusters"]:
            mask = cl["cluster_labels"] == c
            n_c = int(mask.sum())
            if mask.any():
                period_counts = pd.Series(period_anom[mask]).value_counts()
                dominant_period = period_counts.idxmax() if not period_counts.empty else "-"
                c_mean = feat_anom.iloc[mask].mean()
                top_feat = ((c_mean - global_mean) / global_std).abs().idxmax()
            else:
                dominant_period, top_feat = "-", "-"

            pnn_rate = float(preds_anom["pnn"][mask].mean()) if mask.any() else 0.0
            ocsvm_rate = float(preds_anom["transformer_ocsvm"][mask].mean()) if mask.any() else 0.0
            prae_rate = float(preds_anom["prae"][mask].mean()) if mask.any() else 0.0

            if pnn_rate >= 0.5:
                hint = "spoofing-type"
            elif ocsvm_rate >= 0.5 and prae_rate >= 0.5:
                hint = "general (OC-SVM + PRAE)"
            elif ocsvm_rate >= 0.5:
                hint = "general (OC-SVM)"
            elif prae_rate >= 0.5:
                hint = "general (PRAE)"
            else:
                hint = "mixed"

            summary_rows.append({
                "Cluster": cl["cluster_names"][c],
                "N": n_c,
                "Share (%)": round(100 * n_c / n_anom, 2),
                "OC-SVM det.": round(ocsvm_rate, 3),
                "PNN det.": round(pnn_rate, 3),
                "PRAE det.": round(prae_rate, 3),
                "Dominant period": dominant_period,
                "Top feature": top_feat,
                "Interpretation": hint,
            })

        summary_df = pd.DataFrame(summary_rows).set_index("Cluster")
        logger.info("Summary for %d:\n%s", year, summary_df.to_string())
        summary_df.to_csv(os.path.join(output_dir, f"summary_{year}.csv"))

    # ── Plots ────────────────────────────────────────────────────────────
    save_cluster_pca_plot(args.train_years, score_data, cluster_data, anom_data, output_dir, args.random_state)
    save_detection_rate_heatmap(args.train_years, score_data, cluster_data, anom_data, output_dir)

    # ── Save raw results ─────────────────────────────────────────────────
    for year in args.train_years:
        np.savez_compressed(
            os.path.join(output_dir, f"cluster_labels_{year}.npz"),
            labels=cluster_data[year]["cluster_labels"],
            anom_indices=anom_data[year]["anom_indices"],
        )

    logger.info("=" * 70)
    logger.info("Anomaly clustering complete. Output: %s", output_dir)


if __name__ == "__main__":
    main()
