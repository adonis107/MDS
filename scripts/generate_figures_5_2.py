"""
Generate all figures for Section 5.2 (Anomaly Detection on Test Data).

Reads pre-computed CSVs and NPY arrays from results/<year>/test_output/.
Outputs PDF figures to figures/results/.
Run from repo root:  python scripts/generate_figures_5_2.py
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

# ────────────────────────────────────────────────────────────────────
# Output directory
# ────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join("figures", "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────
# Plot style (matches Section 5.1)
# ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})
BLUE   = "#4878CF"
ORANGE = "#E8884A"
RED    = "#C44E52"
GREEN  = "#6AB187"
PURPLE = "#9B59B6"

MODEL_COLORS = {
    "transformer_ocsvm": BLUE,
    "pnn": ORANGE,
    "prae": GREEN,
}
MODEL_LABELS = {
    "transformer_ocsvm": "TF–OC-SVM",
    "pnn": "PNN",
    "prae": "PRAE",
}
PERIOD_ORDER = ["1st_hour", "rest_of_morning", "afternoon", "american_open"]
PERIOD_LABELS = {
    "1st_hour": "1st hour",
    "rest_of_morning": "Rest of morning",
    "afternoon": "Afternoon",
    "american_open": "US open",
}

YEARS = ["2015", "2017"]

def data_dir(year="2015"):
    return os.path.join("results", year, "test_output")

def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  saved {path}")


# ────────────────────────────────────────────────────────────────────
# Figure 1: Anomaly rate by period (grouped bar chart)
# ────────────────────────────────────────────────────────────────────
def fig_anomaly_rates_by_period(year="2015"):
    df = pd.read_csv(os.path.join(data_dir(year), "anomaly_rates_by_period.csv"))
    models = ["transformer_ocsvm", "pnn", "prae"]
    n_periods = len(PERIOD_ORDER)
    x = np.arange(n_periods)
    width = 0.25

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for i, m in enumerate(models):
        sub = df[df["model"] == m].set_index("period").reindex(PERIOD_ORDER)
        bars = ax.bar(x + i * width, sub["rate_pct"], width,
                      label=MODEL_LABELS[m], color=MODEL_COLORS[m], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, sub["rate_pct"]):
            if val > 0.1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)
            elif val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels([PERIOD_LABELS[p] for p in PERIOD_ORDER])
    ax.set_ylabel("Anomaly rate (%)")
    ax.set_title(f"Anomaly rate by time-of-day period ({year} model)")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, f"anomaly_rates_by_period_{year}.pdf")


# ────────────────────────────────────────────────────────────────────
# Figure 2: Daily anomaly rate line chart
# ────────────────────────────────────────────────────────────────────
def fig_daily_anomaly_rates(year="2015"):
    df = pd.read_csv(os.path.join(data_dir(year), "anomaly_rates_by_day.csv"))
    models = ["transformer_ocsvm", "pnn", "prae"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, m in zip(axes, models):
        sub = df[df["model"] == m].copy()
        sub["date"] = pd.to_datetime(sub["day"].str[:10])
        sub = sub.sort_values("date")

        colors = sub["split"].map({
            "out_of_sample": "gray",
            "test_proximate": RED,
            "test_distal": PURPLE,
        })
        ax.scatter(sub["date"], sub["rate_pct"], s=4, c=colors, alpha=0.7, rasterized=True)
        # rolling mean over out_of_sample
        oos = sub[sub["split"] == "out_of_sample"].set_index("date")["rate_pct"]
        rolling = oos.rolling(10, min_periods=3).mean()
        ax.plot(rolling.index, rolling.values, color=MODEL_COLORS[m], linewidth=1.2, label="10-day MA")

        ax.set_ylabel("Rate (%)")
        ax.set_title(MODEL_LABELS[m], fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Date")
    fig.suptitle(f"Daily anomaly rate across test period ({year} model)", fontsize=11, y=1.01)
    fig.tight_layout()

    # add legend for split colours
    handles = [
        Patch(facecolor="gray", label="Out-of-sample"),
        Patch(facecolor=RED, label=r"$\mathcal{T}_A$ (proximate)"),
        Patch(facecolor=PURPLE, label=r"$\mathcal{T}_B$ (distal)"),
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=7, loc="upper left", ncol=3)
    save_fig(fig, f"daily_anomaly_rates_{year}.pdf")


# ────────────────────────────────────────────────────────────────────
# Figure 3: Score distributions (TF-OC-SVM and PRAE only; PNN has no scores)
# ────────────────────────────────────────────────────────────────────
def fig_score_distributions(year="2015"):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

    # TF-OC-SVM
    scores = np.load(os.path.join(data_dir(year), "transformer_ocsvm_scores.npy"))
    preds  = np.load(os.path.join(data_dir(year), "transformer_ocsvm_preds.npy"))
    # subsample for histogramming
    rng = np.random.default_rng(42)
    idx = rng.choice(len(scores), size=min(500_000, len(scores)), replace=False)
    ax = axes[0]
    ax.hist(scores[idx], bins=200, color=BLUE, alpha=0.7, density=True, rasterized=True)
    # mark threshold = 0
    ax.axvline(0, color=RED, linestyle="--", linewidth=1, label="Threshold (0)")
    ax.set_xlabel("OC-SVM decision score")
    ax.set_ylabel("Density")
    ax.set_title("TF–OC-SVM score distribution")
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # PRAE
    scores_prae = np.load(os.path.join(data_dir(year), "prae_scores.npy"))
    preds_prae  = np.load(os.path.join(data_dir(year), "prae_preds.npy"))
    ax = axes[1]
    # clip extreme values for visualisation
    clipped = np.clip(scores_prae[idx], 0, np.percentile(scores_prae[idx], 99.5))
    ax.hist(clipped, bins=200, color=GREEN, alpha=0.7, density=True, rasterized=True)
    ax.set_xlabel("PRAE reconstruction score")
    ax.set_ylabel("Density")
    ax.set_title("PRAE score distribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, f"score_distributions_{year}.pdf")


# ────────────────────────────────────────────────────────────────────
# Figure 4: Proximate vs. distal anomaly rates (bar chart)
# ────────────────────────────────────────────────────────────────────
def fig_proximity_comparison(year="2015"):
    df = pd.read_csv(os.path.join(data_dir(year), "metrics_by_split.csv"))
    models = ["transformer_ocsvm", "pnn", "prae"]
    splits = ["test_proximate", "test_distal"]
    split_labels = {
        "test_proximate": r"$\mathcal{T}_A$ (proximate)",
        "test_distal": r"$\mathcal{T}_B$ (distal)",
    }
    x = np.arange(len(models))
    width = 0.3

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for i, s in enumerate(splits):
        sub = df[df["split"] == s].set_index("model").reindex(models)
        bars = ax.bar(x + i * width, sub["anomaly_rate_pct"], width,
                      label=split_labels[s],
                      color=[RED, PURPLE][i], alpha=0.8, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, sub["anomaly_rate_pct"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}" if val < 0.1 else f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylabel("Anomaly rate (%)")
    ax.set_title(f"Anomaly rate: $\mathcal{{T}}_A$ vs. $\mathcal{{T}}_B$ ({year} model)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, f"proximity_comparison_{year}.pdf")


# ────────────────────────────────────────────────────────────────────
# Figure 5: Consensus agreement (stacked bar / pie)
# ────────────────────────────────────────────────────────────────────
def fig_consensus_agreement(year="2015"):
    df = pd.read_csv(os.path.join(data_dir(year), "consensus_agreement.csv"))
    labels = [f"{int(row.n_models_agreeing)} model{'s' if row.n_models_agreeing != 1 else ''}"
              for _, row in df.iterrows()]
    labels[0] = "No anomaly"
    colors_list = ["#CCCCCC", BLUE, ORANGE, RED]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5),
                             gridspec_kw={"width_ratios": [1, 1.8]})

    # Pie chart (all)
    ax = axes[0]
    ax.pie(df["pct"], labels=labels, colors=colors_list,
           autopct=lambda p: f"{p:.2f}%" if p > 0.001 else "",
           startangle=90, textprops={"fontsize": 8})
    ax.set_title("Overall consensus", fontsize=10)

    # Bar chart (anomalies only, i.e. ≥1 model)
    ax = axes[1]
    anom = df[df["n_models_agreeing"] > 0].copy()
    bars = ax.barh(
        [f"{int(r.n_models_agreeing)} model{'s' if r.n_models_agreeing != 1 else ''}"
         for _, r in anom.iterrows()],
        anom["sample_count"],
        color=[BLUE, ORANGE, RED][:len(anom)],
        edgecolor="white", linewidth=0.5,
    )
    for bar, cnt in zip(bars, anom["sample_count"]):
        ax.text(bar.get_width() + bar.get_width() * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{int(cnt):,}", va="center", fontsize=8)
    ax.set_xlabel("Sample count")
    ax.set_title("Anomalies by number of agreeing models", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, f"consensus_agreement_{year}.pdf")


# ────────────────────────────────────────────────────────────────────
# Figure 6: Score time series for a representative test day
# ────────────────────────────────────────────────────────────────────
def fig_score_timeseries(year="2015"):
    """Plot TF-OC-SVM and PRAE score time series for one proximate test day."""
    with open(os.path.join(data_dir(year), "test_meta.json")) as f:
        meta = json.load(f)

    day_names = meta["day_names"]
    day_labels = meta["day_split_labels"]
    boundaries = meta["day_boundaries"]

    # pick first proximate day
    prox_indices = [i for i, l in enumerate(day_labels) if l == "test_proximate"]
    chosen = prox_indices[0]
    start = boundaries[chosen]
    end = boundaries[chosen + 1]
    day_str = day_names[chosen][:10]

    tf_scores = np.load(os.path.join(data_dir(year), "transformer_ocsvm_scores.npy"),
                        mmap_mode="r")[start:end]
    tf_preds  = np.load(os.path.join(data_dir(year), "transformer_ocsvm_preds.npy"),
                        mmap_mode="r")[start:end]
    prae_scores = np.load(os.path.join(data_dir(year), "prae_scores.npy"),
                          mmap_mode="r")[start:end]
    prae_preds  = np.load(os.path.join(data_dir(year), "prae_preds.npy"),
                          mmap_mode="r")[start:end]
    pnn_preds   = np.load(os.path.join(data_dir(year), "pnn_preds.npy"),
                          mmap_mode="r")[start:end]

    n = len(tf_scores)
    # subsample for plotting (every kth)
    k = max(1, n // 10_000)
    idx = np.arange(0, n, k)
    x = idx / n  # normalised [0,1]

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    # TF-OC-SVM
    ax = axes[0]
    ax.plot(x, np.array(tf_scores)[idx], color=BLUE, linewidth=0.4, alpha=0.8, rasterized=True)
    anom_mask = np.array(tf_preds)[idx] == 1
    if anom_mask.any():
        ax.scatter(x[anom_mask], np.array(tf_scores)[idx][anom_mask],
                   color=RED, s=3, zorder=3, rasterized=True, label="Anomaly")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    ax.set_ylabel("Decision score")
    ax.set_title(f"TF–OC-SVM — {day_str}", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # PNN (binary only)
    ax = axes[1]
    pnn_sub = np.array(pnn_preds)[idx]
    ax.fill_between(x, 0, pnn_sub, color=ORANGE, alpha=0.5, step="mid", rasterized=True)
    ax.set_ylabel("Prediction")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_title(f"PNN — {day_str}", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # PRAE
    ax = axes[2]
    prae_sub = np.array(prae_scores)[idx]
    ax.plot(x, prae_sub, color=GREEN, linewidth=0.4, alpha=0.8, rasterized=True)
    anom_mask_prae = np.array(prae_preds)[idx] == 1
    if anom_mask_prae.any():
        ax.scatter(x[anom_mask_prae], prae_sub[anom_mask_prae],
                   color=RED, s=3, zorder=3, rasterized=True, label="Anomaly")
    ax.set_ylabel("Reconstruction score")
    ax.set_xlabel("Normalised time (0 = open, 1 = close)")
    ax.set_title(f"PRAE — {day_str}", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(f"Score time series on test day {day_str}", fontsize=11, y=1.01)
    fig.tight_layout()
    save_fig(fig, f"score_timeseries_{year}.pdf")


# ────────────────────────────────────────────────────────────────────
# Figure 7: Root-cause top features (horizontal bar chart)
# ────────────────────────────────────────────────────────────────────
def fig_root_cause(year="2015"):
    df = pd.read_csv(os.path.join(data_dir(year), "root_cause_analysis.csv"))
    models = ["transformer_ocsvm", "pnn", "prae"]
    analyses = df["analysis"].unique()

    # Use top10pct_mean for a more representative picture
    analysis = "top10pct_mean" if "top10pct_mean" in analyses else analyses[0]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, m in zip(axes, models):
        sub = df[(df["model"] == m) & (df["analysis"] == analysis)].sort_values("z_score", ascending=True)
        if len(sub) == 0:
            sub = df[(df["model"] == m)].head(15).sort_values("z_score", ascending=True)
        top = sub.tail(10)
        ax.barh(top["feature"], top["z_score"], color=MODEL_COLORS[m], edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Mean z-score")
        ax.set_title(MODEL_LABELS[m], fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle(f"Top features by mean z-score ({analysis.replace('_', ' ')})", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, f"root_cause_features_{year}.pdf")


# ────────────────────────────────────────────────────────────────────
# Figure 8: Anomaly rate heatmap (model × split)
# ────────────────────────────────────────────────────────────────────
def fig_rate_heatmap(year="2015"):
    df = pd.read_csv(os.path.join(data_dir(year), "anomaly_rates_by_day.csv"))
    models = ["transformer_ocsvm", "pnn", "prae"]
    splits = ["out_of_sample", "test_proximate", "test_distal"]
    split_pretty = {"out_of_sample": "Out-of-sample", "test_proximate": r"$\mathcal{T}_A$",
                    "test_distal": r"$\mathcal{T}_B$"}

    matrix = np.zeros((len(models), len(splits)))
    for i, m in enumerate(models):
        for j, s in enumerate(splits):
            sub = df[(df["model"] == m) & (df["split"] == s)]
            matrix[i, j] = sub["rate_pct"].mean() if len(sub) > 0 else 0

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels([split_pretty[s] for s in splits])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in models])
    for i in range(len(models)):
        for j in range(len(splits)):
            txt = f"{matrix[i, j]:.3f}" if matrix[i, j] < 0.1 else f"{matrix[i, j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] > matrix.max() * 0.6 else "black")
    ax.set_title("Mean daily anomaly rate (%) by model and split")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Rate (%)")
    fig.tight_layout()
    save_fig(fig, f"rate_heatmap_{year}.pdf")


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Section 5.2 figures for both years...")
    for year in YEARS:
        print(f"\n=== Year {year} ===")
        fig_anomaly_rates_by_period(year)
        fig_daily_anomaly_rates(year)
        fig_score_distributions(year)
        fig_proximity_comparison(year)
        fig_consensus_agreement(year)
        fig_score_timeseries(year)
        fig_root_cause(year)
        fig_rate_heatmap(year)
    print("\nDone.")
