"""
Generate figures for Section 5.5 (Spoofing Gain Analysis).

The PNN "scores" saved in pnn_scores.npy are the spoofing gain G
(expected cost reduction from spoofing).  G > 0 means profitable spoofing
is predicted.  This script analyses G's distribution and its relationship
to each model's anomaly score and detection flag.

Run from repo root:  python scripts/generate_figures_5_5.py
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Output directory ────────────────────────────────────────────────
OUT_DIR = os.path.join("figures", "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Plot style ──────────────────────────────────────────────────────
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
GREEN  = "#6AB187"
RED    = "#C44E52"
GREY   = "#999999"

MODEL_COLORS = {
    "transformer_ocsvm": BLUE,
    "pnn": ORANGE,
    "prae": GREEN,
}
MODEL_LABELS = {
    "transformer_ocsvm": "TF\u2013OC-SVM",
    "pnn": "PNN",
    "prae": "PRAE",
}
MODELS = ["transformer_ocsvm", "pnn", "prae"]
YEARS  = ["2015", "2017"]


def data_dir(year):
    return os.path.join("results", year, "test_output")


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  saved {path}")


def load_data(year):
    """Return dict with gain G, scores, and preds for each model."""
    G = np.load(os.path.join(data_dir(year), "pnn_scores.npy"))
    fin = np.isfinite(G)
    out = {"G": G, "fin": fin}
    for m in MODELS:
        out[f"{m}_scores"] = np.load(os.path.join(data_dir(year), f"{m}_scores.npy"))
        out[f"{m}_preds"]  = np.load(os.path.join(data_dir(year), f"{m}_preds.npy"))
    return out


# ── Figure 5.5.1: Gain distribution ────────────────────────────────
def fig_gain_distribution(data_by_year):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    for ax, year in zip(axes, YEARS):
        d = data_by_year[year]
        G = d["G"][d["fin"]]

        # Clip extreme tails for visualization
        lo, hi = np.percentile(G, [0.5, 99.5])
        G_clip = G[(G >= lo) & (G <= hi)]

        if G_clip.std() < 1e-6:
            # Degenerate distribution (2015): show a bar at zero
            ax.hist(G_clip, bins=100, color=BLUE, alpha=0.7, edgecolor="white",
                    linewidth=0.3)
            ax.set_xlabel("Spoofing gain $G$")
            ax.set_ylabel("Count")
        else:
            ax.hist(G_clip, bins=200, color=BLUE, alpha=0.7, edgecolor="white",
                    linewidth=0.3, log=True)
            ax.set_xlabel("Spoofing gain $G$")
            ax.set_ylabel("Count (log scale)")

        ax.axvline(0, color=RED, linestyle="--", linewidth=1, label="$G = 0$")
        frac_pos = (d["G"][d["fin"]] > 0).mean() * 100
        ax.set_title(f"{year}   ($G > 0$: {frac_pos:.2f}%)")
        ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "fig_5_5_gain_distribution.pdf")


# ── Figure 5.5.2: Gain vs. anomaly score scatter ───────────────────
def fig_gain_vs_score(data_by_year):
    # Only TF-OCSVM and PRAE (PNN score IS the gain)
    score_models = ["transformer_ocsvm", "prae"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    rng = np.random.default_rng(42)

    for col, year in enumerate(YEARS):
        d = data_by_year[year]
        fin = d["fin"]
        G = d["G"][fin]

        for row, m in enumerate(score_models):
            ax = axes[row, col]
            s = d[f"{m}_scores"][fin]
            p = d[f"{m}_preds"][fin]

            # Subsample
            n = min(5000, len(G))
            idx = rng.choice(len(G), n, replace=False)
            Gs, ss, ps = G[idx], s[idx], p[idx]

            # Plot non-flagged first (grey), then flagged (red)
            nf = ps == 0
            ax.scatter(Gs[nf], ss[nf], s=4, c=GREY, alpha=0.3, rasterized=True)
            fl = ps == 1
            if fl.any():
                ax.scatter(Gs[fl], ss[fl], s=8, c=RED, alpha=0.6,
                           edgecolors="white", linewidths=0.3, rasterized=True)

            # Spearman on this subsample
            rho, _ = sp_stats.spearmanr(Gs, ss)
            ax.set_title(f"{MODEL_LABELS[m]}, {year}  ($\\rho_s = {rho:.3f}$)")
            ax.set_xlabel("Spoofing gain $G$")
            ax.set_ylabel("Anomaly score")

    # Custom legend
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY,
                       markersize=5, label="Not flagged"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor=RED,
                       markersize=5, label="Flagged")]
    axes[0, 1].legend(handles=handles, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "fig_5_5_gain_vs_score.pdf")


# ── Figure 5.5.3: Gain KDE, flagged vs. non-flagged ────────────────
def fig_gain_flagged_kde(data_by_year):
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    rng = np.random.default_rng(42)

    for col, year in enumerate(YEARS):
        d = data_by_year[year]
        fin = d["fin"]
        G = d["G"][fin]

        for row, m in enumerate(MODELS):
            ax = axes[row, col]
            p = d[f"{m}_preds"][fin]
            flagged = G[p == 1]
            nonflagged = G[p == 0]

            # Subsample for KDE
            n_kde = 20000
            if len(flagged) > n_kde:
                flagged = flagged[rng.choice(len(flagged), n_kde, replace=False)]
            if len(nonflagged) > n_kde:
                nonflagged = nonflagged[rng.choice(len(nonflagged), n_kde, replace=False)]

            # Clip to common range for visualization
            lo = min(np.percentile(flagged, 1), np.percentile(nonflagged, 1))
            hi = max(np.percentile(flagged, 99), np.percentile(nonflagged, 99))
            if hi - lo < 1e-6:
                hi = lo + 1e-3  # avoid degenerate range

            bins = np.linspace(lo, hi, 120)
            ax.hist(nonflagged, bins=bins, density=True, alpha=0.5, color=GREY,
                    label="Not flagged", edgecolor="white", linewidth=0.3)
            ax.hist(flagged, bins=bins, density=True, alpha=0.6,
                    color=MODEL_COLORS[m], label="Flagged",
                    edgecolor="white", linewidth=0.3)

            ax.axvline(0, color=RED, linestyle="--", linewidth=0.8)
            ax.set_title(f"{MODEL_LABELS[m]}, {year}")
            ax.set_xlabel("Spoofing gain $G$")
            ax.set_ylabel("Density")
            ax.legend(loc="upper left", framealpha=0.9, fontsize=7)

    fig.tight_layout()
    save_fig(fig, "fig_5_5_gain_flagged_kde.pdf")


# ── Figure 5.5.4: Mean gain by score decile ────────────────────────
def fig_gain_by_decile(data_by_year):
    # Only TF-OCSVM and PRAE (PNN score = gain, so trivially monotone)
    score_models = ["transformer_ocsvm", "prae"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    rng = np.random.default_rng(42)

    for col, year in enumerate(YEARS):
        d = data_by_year[year]
        fin = d["fin"]
        G = d["G"][fin]

        for row, m in enumerate(score_models):
            ax = axes[row, col]
            s = d[f"{m}_scores"][fin]

            # Subsample for speed
            n_sub = min(500000, len(G))
            idx = rng.choice(len(G), n_sub, replace=False)
            Gs, ss = G[idx], s[idx]

            # Compute deciles
            decile_edges = np.percentile(ss, np.arange(0, 110, 10))
            decile_labels = np.digitize(ss, decile_edges[1:-1])

            means, ses = [], []
            for dec in range(10):
                mask = decile_labels == dec
                vals = Gs[mask]
                means.append(vals.mean())
                ses.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0)

            x = np.arange(1, 11)
            ax.bar(x, means, yerr=ses, capsize=3, color=MODEL_COLORS[m],
                   edgecolor="white", linewidth=0.5, alpha=0.8)
            ax.axhline(0, color="grey", linestyle="--", linewidth=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels([f"D{i}" for i in x], fontsize=8)
            ax.set_xlabel("Score decile (low to high)")
            ax.set_ylabel("Mean $G$")
            ax.set_title(f"{MODEL_LABELS[m]}, {year}")

    fig.tight_layout()
    save_fig(fig, "fig_5_5_gain_by_decile.pdf")


# ── Compute statistics table ────────────────────────────────────────
def compute_stats(data_by_year):
    """Compute Spearman correlations and Mann-Whitney tests."""
    rng = np.random.default_rng(42)
    rows = []

    for year in YEARS:
        d = data_by_year[year]
        fin = d["fin"]
        G = d["G"][fin]

        for m in MODELS:
            s = d[f"{m}_scores"][fin]
            p = d[f"{m}_preds"][fin]

            # Spearman (subsample)
            if m == "pnn":
                rho, p_rho = 1.0, 0.0  # score IS the gain
            else:
                n_sub = min(100000, len(G))
                idx = rng.choice(len(G), n_sub, replace=False)
                rho, p_rho = sp_stats.spearmanr(s[idx], G[idx])

            # Mann-Whitney: G in flagged vs non-flagged
            flagged_G = G[p == 1]
            nonflagged_G = G[p == 0]
            n_f, n_nf = len(flagged_G), len(nonflagged_G)

            f_sub = flagged_G if n_f <= 20000 else flagged_G[rng.choice(n_f, 20000, replace=False)]
            nf_sub = nonflagged_G if n_nf <= 20000 else nonflagged_G[rng.choice(n_nf, 20000, replace=False)]
            U, p_U = sp_stats.mannwhitneyu(f_sub, nf_sub, alternative="two-sided")
            rbc = 1.0 - (2.0 * U) / (len(f_sub) * len(nf_sub))

            rows.append({
                "year": year, "model": m,
                "spearman_rho": float(rho), "spearman_p": float(p_rho),
                "mean_G_flagged": float(flagged_G.mean()) if n_f > 0 else float("nan"),
                "mean_G_nonflagged": float(nonflagged_G.mean()),
                "mw_rbc": float(rbc), "mw_p": float(p_U),
                "n_flagged": n_f, "n_nonflagged": n_nf,
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "spoofing_gain_stats.csv"), index=False)
    print(f"  saved {os.path.join(OUT_DIR, 'spoofing_gain_stats.csv')}")
    print(df.to_string(index=False))
    return df


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    data_by_year = {year: load_data(year) for year in YEARS}
    print()

    print("Computing statistics...")
    stats_df = compute_stats(data_by_year)
    print()

    print("Generating figures...")
    fig_gain_distribution(data_by_year)
    fig_gain_vs_score(data_by_year)
    fig_gain_flagged_kde(data_by_year)
    fig_gain_by_decile(data_by_year)

    print("\nDone.")
