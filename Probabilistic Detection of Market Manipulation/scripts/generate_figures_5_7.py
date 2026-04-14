"""
Compute anomaly clustering and generate figures for Section 5.7.

Clusters anomalous windows (flagged by â‰¥1 model) in the 3-D normalised
score space using HDBSCAN, following the methodology of Section 3.6.1.
Saves cluster labels, statistics, and generates three figures.

Run from repo root:  python scripts/generate_figures_5_7.py
"""

import os, sys, json, warnings, gc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier

sys.path.insert(0, os.path.abspath("."))

from detection.data.loaders import create_sequences
from detection.sensitivity.occlusion import parse_feature_attributes

OUT_DIR = os.path.join("figures", "results")
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LENGTH = 25
HDBSCAN_MIN_SAMPLES = 15
HDBSCAN_FRAC = 20
HDBSCAN_MAX_MIN_SIZE = 500
MAX_HDBSCAN_POINTS = 20000
N_PROFILE_PER_CLUSTER = 300
YEARS = ["2015", "2017"]
MODEL_TYPES = ["transformer_ocsvm", "pnn", "prae"]

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = (
    "#4878CF", "#E8884A", "#6AB187", "#C44E52", "#999999", "#9B59B6")

CLUSTER_PALETTE = [
    "#4878CF", "#E8884A", "#6AB187", "#C44E52", "#9B59B6",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]
NOISE_COLOR = "#D0D0D0"

TYPE_COLORS = {
    "hawkes": "#E24A33", "ofi": "#348ABD", "imbalance": "#988ED5",
    "dynamics": "#FBC15E", "flow": "#8EBA42", "trade": "#FFB5B8",
    "cancel": "#777777", "sma": "#AAAAAA", "lob_price": "#55A868",
    "lob_volume": "#64B5CD", "spread": "#C44E52", "mid_price": "#4878CF",
    "depth": "#9B59B6", "volatility": "#E8884A", "return": "#6AB187",
    "sweep": "#D6616B", "slope": "#CE8DBE", "rapidity": "#8C6D31",
    "speed": "#BD9E39", "deep_order": "#E7969C", "time": "#CCCCCC",
    "other": "#DDDDDD",
}


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  saved {path}")


def load_feature_names(year):
    with open(f"results/{year}/pnn_features.txt") as f:
        return [line.strip() for line in f if line.strip()]


def get_feature_types(feature_names):
    return [parse_feature_attributes(fn)["type"] for fn in feature_names]


def _load_test_meta(year):
    with open(f"results/{year}/test_output/test_meta.json") as f:
        return json.load(f)


def _test_day_indices(meta):
    return [i for i, l in enumerate(meta["day_split_labels"]) if "test" in l]


def cluster_year(year):
    """
    Load scores/preds, select anomalous windows, cluster in 3-D score space.
    Returns dict with all clustering results.
    """
    meta = _load_test_meta(year)
    bounds = meta["day_boundaries"]
    day_names = meta["day_names"]
    test_days = _test_day_indices(meta)
    test_start = bounds[test_days[0]]
    test_end = bounds[test_days[-1] + 1]
    n_test = test_end - test_start

    scores_raw = {}
    preds = {}
    for mt in MODEL_TYPES:
        s = np.load(f"results/{year}/test_output/{mt}_scores.npy")[test_start:test_end]
        p = np.load(f"results/{year}/test_output/{mt}_preds.npy")[test_start:test_end]
        nan_mask = ~np.isfinite(s)
        if nan_mask.any():
            s[nan_mask] = np.nanmin(s) if np.any(np.isfinite(s)) else 0.0
        scores_raw[mt] = s
        preds[mt] = p
    print(f"  Test windows: {n_test}")

    n_models_flagged = sum(preds[mt].astype(int) for mt in MODEL_TYPES)
    anom_mask = n_models_flagged >= 1
    anom_indices = np.where(anom_mask)[0]
    n_anom = len(anom_indices)
    print(f"  Anomalous windows (>=1 model): {n_anom} ({100*n_anom/n_test:.2f}%)")

    for k in [1, 2, 3]:
        ck = (n_models_flagged == k).sum()
        print(f"    {k}-model agreement: {ck}")

    score_matrix_full = np.column_stack([scores_raw[mt] for mt in MODEL_TYPES])
    scaler = MinMaxScaler()
    score_matrix_full_norm = scaler.fit_transform(score_matrix_full)
    X_anom = score_matrix_full_norm[anom_indices]

    rng_clust = np.random.default_rng(42)
    min_cluster_size = max(5, min(HDBSCAN_MAX_MIN_SIZE, n_anom // HDBSCAN_FRAC))

    if n_anom > MAX_HDBSCAN_POINTS:
        sub_idx = rng_clust.choice(n_anom, size=MAX_HDBSCAN_POINTS, replace=False)
        X_sub = X_anom[sub_idx]
        min_cs_sub = max(5, min(HDBSCAN_MAX_MIN_SIZE, MAX_HDBSCAN_POINTS // HDBSCAN_FRAC))
        print(f"  HDBSCAN on {MAX_HDBSCAN_POINTS} subsample: min_cluster_size={min_cs_sub}, "
              f"min_samples={HDBSCAN_MIN_SAMPLES}")
        hdb = HDBSCAN(min_cluster_size=min_cs_sub, min_samples=HDBSCAN_MIN_SAMPLES)
        sub_labels = hdb.fit_predict(X_sub)
        print(f"  Subsample clustering done: {len(set(sub_labels)) - (1 if -1 in sub_labels else 0)} clusters")

        valid_sub = sub_labels != -1
        if valid_sub.sum() >= 5:
            knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
            knn.fit(X_sub[valid_sub], sub_labels[valid_sub])
            print(f"  KNN propagation to {n_anom} points...", flush=True)
            batch_size = 100000
            cluster_labels = np.empty(n_anom, dtype=int)
            for i in range(0, n_anom, batch_size):
                cluster_labels[i:i+batch_size] = knn.predict(X_anom[i:i+batch_size])
            noise_frac = (~valid_sub).sum() / len(sub_labels)
            if noise_frac > 0.005:
                dists, _ = knn.kneighbors(X_anom, n_neighbors=1)
                dist_threshold = np.percentile(dists.ravel(), 100 * (1 - noise_frac))
                cluster_labels[dists.ravel() > dist_threshold] = -1
            print(f"  KNN propagation done.", flush=True)
        else:
            cluster_labels = np.full(n_anom, -1, dtype=int)
    else:
        print(f"  HDBSCAN: min_cluster_size={min_cluster_size}, min_samples={HDBSCAN_MIN_SAMPLES}")
        hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=HDBSCAN_MIN_SAMPLES)
        cluster_labels = hdb.fit_predict(X_anom)

    print(f"  Label propagation done.", flush=True)
    unique_clusters = sorted(set(cluster_labels))
    n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    n_noise = int((cluster_labels == -1).sum())
    print(f"  Clusters: {n_clusters}, noise: {n_noise} ({100*n_noise/n_anom:.1f}%)")

    valid_mask = cluster_labels != -1
    n_valid = int(valid_mask.sum())
    if n_valid >= 2 and n_clusters >= 2:
        if n_valid > 20000:
            sil_idx = rng_clust.choice(np.where(valid_mask)[0], size=20000, replace=False)
            sil = silhouette_score(X_anom[sil_idx], cluster_labels[sil_idx])
            db = davies_bouldin_score(X_anom[sil_idx], cluster_labels[sil_idx])
        else:
            sil = silhouette_score(X_anom[valid_mask], cluster_labels[valid_mask])
            db = davies_bouldin_score(X_anom[valid_mask], cluster_labels[valid_mask])
    else:
        sil, db = np.nan, np.nan
    print(f"  Silhouette: {sil:.4f}, Davies-Bouldin: {db:.4f}")

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_anom)

    cluster_stats = []
    for c in unique_clusters:
        mask = cluster_labels == c
        n_c = int(mask.sum())
        pnn_gain = scores_raw["pnn"][anom_indices[mask]]
        mean_gain = float(np.nanmean(pnn_gain))
        det_rates = {mt: float(preds[mt][anom_indices[mask]].mean()) for mt in MODEL_TYPES}
        global_indices = anom_indices[mask] + test_start
        day_idx = np.searchsorted(bounds, global_indices, side="right") - 1
        cluster_stats.append({
            "cluster": c,
            "n": n_c,
            "pct": round(100 * n_c / n_anom, 2),
            "mean_gain": round(mean_gain, 4),
            "ocsvm_rate": round(det_rates["transformer_ocsvm"], 3),
            "pnn_rate": round(det_rates["pnn"], 3),
            "prae_rate": round(det_rates["prae"], 3),
            "mean_ocsvm_norm": round(float(X_anom[mask, 0].mean()), 4),
            "mean_pnn_norm": round(float(X_anom[mask, 1].mean()), 4),
            "mean_prae_norm": round(float(X_anom[mask, 2].mean()), 4),
        })
    stats_df = pd.DataFrame(cluster_stats)

    return {
        "meta": meta,
        "bounds": bounds,
        "day_names": day_names,
        "test_days": test_days,
        "test_start": test_start,
        "test_end": test_end,
        "n_test": n_test,
        "n_anom": n_anom,
        "anom_indices": anom_indices,
        "X_anom": X_anom,
        "X_pca": X_pca,
        "pca": pca,
        "cluster_labels": cluster_labels,
        "unique_clusters": unique_clusters,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "sil": sil,
        "db": db,
        "stats_df": stats_df,
        "scores_raw": scores_raw,
        "preds": preds,
        "n_models_flagged": n_models_flagged[anom_indices],
    }


def compute_feature_group_profile(year, res, feature_names, feature_types, rng):
    """
    Subsample windows per cluster, load their raw features,
    compute mean per feature group.  Returns (group_names, profile_matrix).
    profile_matrix: shape (n_clusters_with_noise, n_groups), standardised.
    """
    meta = res["meta"]
    bounds = res["bounds"]
    day_names = res["day_names"]
    test_start = res["test_start"]
    cluster_labels = res["cluster_labels"]
    anom_indices = res["anom_indices"]
    unique_clusters = res["unique_clusters"]

    group_map = {}
    for i, ft in enumerate(feature_types):
        group_map.setdefault(ft, []).append(i)
    group_names = sorted(group_map.keys())
    n_groups = len(group_names)

    sampled_local = []
    for c in unique_clusters:
        c_positions = np.where(cluster_labels == c)[0]
        n_pick = min(N_PROFILE_PER_CLUSTER, len(c_positions))
        picked = rng.choice(c_positions, size=n_pick, replace=False)
        for p in picked:
            sampled_local.append((c, p))

    global_indices = np.array([anom_indices[p] + test_start for _, p in sampled_local])
    day_idx_arr = np.searchsorted(bounds[:-1], global_indices, side="right") - 1

    day_to_samples = {}
    for si, (_, p) in enumerate(sampled_local):
        di = int(day_idx_arr[si])
        local_row = int(global_indices[si] - bounds[di]) + SEQ_LENGTH - 1
        day_to_samples.setdefault(di, []).append((si, local_row))

    n_sampled = len(sampled_local)
    group_values = np.zeros((n_sampled, n_groups), dtype=np.float32)
    data_dir = "data/processed/TOTF.PA-book"

    for di, samples in sorted(day_to_samples.items()):
        path = os.path.join(data_dir, day_names[di])
        if not os.path.exists(path):
            print(f"    WARNING: {path} not found")
            continue
        try:
            df = pd.read_parquet(path, columns=feature_names)
        except Exception:
            full = pd.read_parquet(path)
            available = [c for c in feature_names if c in full.columns]
            df = full[available].copy()
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[feature_names]
            del full
        arr = df.values.astype(np.float32)
        for si, row_idx in samples:
            if 0 <= row_idx < len(arr):
                row = arr[row_idx]
                for gi, gn in enumerate(group_names):
                    group_values[si, gi] = np.mean(row[group_map[gn]])
        del arr, df
        gc.collect()

    gv_mean = group_values.mean(axis=0, keepdims=True)
    gv_std = group_values.std(axis=0, keepdims=True)
    gv_std[gv_std == 0] = 1.0
    group_values_std = (group_values - gv_mean) / gv_std

    sample_clusters = np.array([c for c, _ in sampled_local])
    profile_rows = {}
    for c in unique_clusters:
        mask = sample_clusters == c
        if mask.sum() > 0:
            profile_rows[c] = group_values_std[mask].mean(axis=0)
        else:
            profile_rows[c] = np.zeros(n_groups)

    profile_matrix = np.array([profile_rows[c] for c in unique_clusters])
    return group_names, profile_matrix


def fig_cluster_projection(results):
    """PCA scatter of flagged windows, coloured by cluster. One panel per year."""
    n_years = len(results)
    fig, axes = plt.subplots(n_years, 1, figsize=(7, 5 * n_years))
    if n_years == 1:
        axes = [axes]

    for ax, (year, res) in zip(axes, results.items()):
        X_pca = res["X_pca"]
        labels = res["cluster_labels"]
        pca = res["pca"]
        unique = res["unique_clusters"]

        MAX_PLOT = 50000
        n_pts = len(X_pca)
        if n_pts > MAX_PLOT:
            plot_idx = np.random.default_rng(42).choice(n_pts, size=MAX_PLOT, replace=False)
            X_pca_plot = X_pca[plot_idx]
            labels_plot = labels[plot_idx]
        else:
            X_pca_plot = X_pca
            labels_plot = labels

        noise_mask = labels_plot == -1
        if noise_mask.any():
            ax.scatter(X_pca_plot[noise_mask, 0], X_pca_plot[noise_mask, 1],
                       s=3, alpha=0.15, c=NOISE_COLOR, rasterized=True)
        centroids = []
        for c in unique:
            if c == -1:
                continue
            mask = labels_plot == c
            color = CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)]
            ax.scatter(X_pca_plot[mask, 0], X_pca_plot[mask, 1],
                       s=6, alpha=0.4, c=color, rasterized=True)
            full_mask = labels == c
            cx, cy = X_pca[full_mask, 0].mean(), X_pca[full_mask, 1].mean()
            centroids.append((c, cx, cy, color))
        for c, cx, cy, color in centroids:
            ax.scatter(cx, cy, s=120, c=color, edgecolors="black",
                       linewidths=1.2, marker="D", zorder=10)
            ax.annotate(f"C{c}", (cx, cy), fontsize=8, fontweight="bold",
                        xytext=(5, 5), textcoords="offset points")

        handles = []
        for c in unique:
            if c == -1:
                handles.append(Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=NOISE_COLOR, markersize=5,
                               label=f"Noise (n={int((labels==-1).sum())})"))
            else:
                mask = labels == c
                handles.append(Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)],
                               markersize=6, label=f"C{c} (n={int(mask.sum())})"))
        ax.legend(handles=handles, fontsize=7, loc="best", markerscale=1.2)
        ev = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({ev[0]:.1%})")
        ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
        ax.set_title(f"{year}: HDBSCAN clustering (k={res['n_clusters']}, "
                     f"sil={res['sil']:.3f})")
    fig.tight_layout()
    save_fig(fig, "fig_5_7_cluster_pca.pdf")


def fig_feature_group_profile(results, profiles):
    """Heatmap: clusters Ã- feature groups, standardised mean values."""
    n_years = len(results)
    fig, axes = plt.subplots(n_years, 1, figsize=(10, 3.5 * n_years + 1))
    if n_years == 1:
        axes = [axes]

    for ax, (year, res) in zip(axes, results.items()):
        group_names, matrix = profiles[year]
        unique = res["unique_clusters"]
        row_labels = [f"C{c}" if c != -1 else "Noise" for c in unique]

        keep = [i for i, c in enumerate(unique) if c != -1]
        if not keep:
            ax.text(0.5, 0.5, "No clusters", transform=ax.transAxes, ha="center")
            ax.set_title(f"{year}")
            continue
        mat = matrix[keep]
        rl = [row_labels[i] for i in keep]

        vmax = max(abs(mat.min()), abs(mat.max()), 0.5)
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(group_names)))
        ax.set_xticklabels(group_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(rl)))
        ax.set_yticklabels(rl, fontsize=9)
        if len(rl) <= 8 and len(group_names) <= 25:
            for i in range(len(rl)):
                for j in range(len(group_names)):
                    v = mat[i, j]
                    if abs(v) > 0.05:
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                fontsize=6,
                                color="white" if abs(v) > vmax * 0.6 else "black")
        fig.colorbar(im, ax=ax, label="Standardised mean", shrink=0.8, pad=0.02)
        ax.set_title(f"{year}: mean standardised feature group value per cluster")

    fig.tight_layout()
    save_fig(fig, "fig_5_7_feature_group_profile.pdf")


def fig_cluster_temporal(results):
    """Bar chart of cluster sizes + timeline scatter per year."""
    n_years = len(results)
    fig, axes = plt.subplots(n_years, 2, figsize=(12, 4 * n_years),
                             gridspec_kw={"width_ratios": [1, 2.5]})
    if n_years == 1:
        axes = axes.reshape(1, -1)

    for row, (year, res) in enumerate(results.items()):
        labels = res["cluster_labels"]
        unique = res["unique_clusters"]
        anom_indices = res["anom_indices"]
        bounds = res["bounds"]
        test_start = res["test_start"]
        test_days = res["test_days"]
        day_names = res["day_names"]

        ax_left = axes[row, 0]
        sizes = [(c, int((labels == c).sum())) for c in unique]
        sizes.sort(key=lambda x: x[1], reverse=True)
        bar_labels = [f"C{c}" if c != -1 else "Noise" for c, _ in sizes]
        bar_vals = [s for _, s in sizes]
        bar_colors = [CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)] if c != -1 else NOISE_COLOR
                      for c, _ in sizes]
        ax_left.barh(range(len(bar_labels)), bar_vals, color=bar_colors,
                     edgecolor="white", linewidth=0.3)
        ax_left.set_yticks(range(len(bar_labels)))
        ax_left.set_yticklabels(bar_labels, fontsize=9)
        ax_left.set_xlabel("Window count")
        ax_left.set_title(f"{year}: cluster sizes")
        ax_left.invert_yaxis()

        ax_right = axes[row, 1]
        global_idx = anom_indices + test_start
        day_idx = np.searchsorted(bounds[:-1], global_idx, side="right") - 1

        test_day_set = set(test_days)
        test_day_order = {d: i for i, d in enumerate(test_days)}

        rng = np.random.default_rng(42)
        for c in unique:
            c_mask = labels == c
            c_pos = np.where(c_mask)[0]
            if len(c_pos) > 5000:
                c_pos = rng.choice(c_pos, size=5000, replace=False)
            c_days = day_idx[c_pos]
            c_day_pos = np.array([test_day_order.get(int(d), -1) for d in c_days])
            valid = c_day_pos >= 0
            if valid.sum() == 0:
                continue
            color = CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)] if c != -1 else NOISE_COLOR
            alpha = 0.15 if c == -1 else 0.35
            y_jitter = rng.normal(c if c != -1 else -1, 0.15, size=valid.sum())
            ax_right.scatter(c_day_pos[valid], y_jitter,
                             s=2, alpha=alpha, c=color, rasterized=True)

        short_names = [day_names[d].split("-TOTF")[0] for d in test_days]
        split_labels = res["meta"]["day_split_labels"]
        ax_right.set_xticks(range(len(test_days)))
        ax_right.set_xticklabels(short_names, rotation=45, ha="right", fontsize=6)
        ax_right.set_ylabel("Cluster")
        ax_right.set_title(f"{year}: temporal distribution of anomalies")

        for i, di in enumerate(test_days):
            if "distal" in split_labels[di]:
                ax_right.axvspan(i - 0.5, i + 0.5, color="#FFE0E0", alpha=0.3, zorder=0)

    fig.tight_layout()
    save_fig(fig, "fig_5_7_cluster_temporal.pdf")


def main():
    rng = np.random.default_rng(42)
    feature_names = load_feature_names("2015")
    feature_types = get_feature_types(feature_names)

    results = {}
    profiles = {}

    for year in YEARS:
        print(f"\n{'='*60}\nYear {year}\n{'='*60}")
        res = cluster_year(year)
        results[year] = res

        out_path = f"results/{year}/test_output/cluster_labels.npy"
        np.save(out_path, res["cluster_labels"])
        print(f"  saved {out_path}")

        stats_path = f"results/{year}/test_output/cluster_stats.csv"
        res["stats_df"].to_csv(stats_path, index=False)
        print(f"  saved {stats_path}")

        print(f"\n--- Feature group profile ({year}) ---")
        group_names, profile_matrix = compute_feature_group_profile(
            year, res, feature_names, feature_types, rng)
        profiles[year] = (group_names, profile_matrix)

    print("\n--- Generating figures ---")
    fig_cluster_projection(results)
    fig_feature_group_profile(results, profiles)
    fig_cluster_temporal(results)

    print("\n\n=== SUMMARY FOR LATEX ===")
    for year in YEARS:
        res = results[year]
        print(f"\n{year}:")
        print(f"  Anomalous windows: {res['n_anom']}")
        print(f"  Clusters: {res['n_clusters']}")
        print(f"  Noise: {res['n_noise']} ({100*res['n_noise']/res['n_anom']:.1f}%)")
        print(f"  Silhouette: {res['sil']:.4f}")
        print(f"  Davies-Bouldin: {res['db']:.4f}")
        print(f"\n  Cluster summary:")
        print(res["stats_df"].to_string(index=False))

        group_names, profile = profiles[year]
        unique = res["unique_clusters"]
        for i, c in enumerate(unique):
            if c == -1:
                continue
            vals = profile[i]
            top2_idx = np.argsort(np.abs(vals))[::-1][:2]
            top2 = [(group_names[j], vals[j]) for j in top2_idx]
            print(f"  C{c}: top groups = {top2[0][0]} ({top2[0][1]:+.2f}), "
                  f"{top2[1][0]} ({top2[1][1]:+.2f})")

    print("\nDone.")


if __name__ == "__main__":
    main()

