"""
Generate figures for Section 5.4 (Proximity Analysis).

Reads anomaly_rates_by_day.csv and proximity_comparison.csv from
results/<year>/test_output/.  Outputs PDF figures to figures/results/.
Run from repo root:  python scripts/generate_figures_5_4.py
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join("figures", "results")
os.makedirs(OUT_DIR, exist_ok=True)

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
YEARS = ["2015", "2017"]
YEAR_MARKERS = {"2015": "o", "2017": "s"}
SPLIT_LABELS = {"test_proximate": r"$\mathcal{T}_A$",
                "test_distal": r"$\mathcal{T}_B$"}


def data_dir(year):
    return os.path.join("results", year, "test_output")


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  saved {path}")


def load_daily_rates():
    """Load per-day anomaly rates for proximate/distal splits, both years."""
    frames = []
    for year in YEARS:
        df = pd.read_csv(os.path.join(data_dir(year), "anomaly_rates_by_day.csv"))
        df = df[df["split"].isin(["test_proximate", "test_distal"])].copy()
        df["year"] = year
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def compute_welch_table(daily):
    """Compute Welch t-test on per-day anomaly rates for each (model, year)."""
    rows = []
    for year in YEARS:
        for model in MODELS:
            sub = daily[(daily["year"] == year) & (daily["model"] == model)]
            ta = sub.loc[sub["split"] == "test_proximate", "rate_pct"].values
            tb = sub.loc[sub["split"] == "test_distal", "rate_pct"].values
            mean_a, mean_b = ta.mean(), tb.mean()
            diff = mean_a - mean_b
            t_stat, p_val = sp_stats.ttest_ind(ta, tb, equal_var=False)
            se = np.sqrt(ta.var(ddof=1)/len(ta) + tb.var(ddof=1)/len(tb))
            rows.append({
                "year": year,
                "model": model,
                "mean_ta": mean_a,
                "mean_tb": mean_b,
                "diff": diff,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_val,
                "n_ta": len(ta),
                "n_tb": len(tb),
            })
    return pd.DataFrame(rows)


def fig_ta_vs_tb_dots(daily):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=False)

    for ax, model in zip(axes, MODELS):
        for year in YEARS:
            sub = daily[(daily["year"] == year) & (daily["model"] == model)]
            ta_rates = sub.loc[sub["split"] == "test_proximate", "rate_pct"].values
            tb_rates = sub.loc[sub["split"] == "test_distal", "rate_pct"].values

            marker = YEAR_MARKERS[year]
            color = MODEL_COLORS[model]
            alpha = 1.0 if year == "2015" else 0.6

            jitter_a = np.random.default_rng(42).uniform(-0.08, 0.08, len(ta_rates))
            jitter_b = np.random.default_rng(43).uniform(-0.08, 0.08, len(tb_rates))
            ax.scatter(np.zeros(len(ta_rates)) + jitter_a, ta_rates,
                       marker=marker, color=color, alpha=alpha, s=40,
                       edgecolors="white", linewidths=0.5, zorder=3)
            ax.scatter(np.ones(len(tb_rates)) + jitter_b, tb_rates,
                       marker=marker, color=color, alpha=alpha, s=40,
                       edgecolors="white", linewidths=0.5, zorder=3)

            mean_a, mean_b = ta_rates.mean(), tb_rates.mean()
            ls = "-" if year == "2015" else "--"
            ax.plot([0, 1], [mean_a, mean_b], ls=ls, color=color,
                    alpha=alpha, linewidth=1.5, zorder=2,
                    label=f"{year}")

        ax.set_xticks([0, 1])
        ax.set_xticklabels([r"$\mathcal{T}_A$", r"$\mathcal{T}_B$"])
        ax.set_title(MODEL_LABELS[model])
        ax.set_ylabel("Anomaly rate (%)" if model == MODELS[0] else "")
        ax.legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "fig_5_4_ta_vs_tb_dots.pdf")


def fig_effect_size_bars(tbl):
    fig, ax = plt.subplots(figsize=(6, 3.5))

    n_models = len(MODELS)
    n_years = len(YEARS)
    group_width = 0.7
    bar_width = group_width / n_years
    x = np.arange(n_models)

    year_colors = {"2015": "#4878CF", "2017": "#E8884A"}

    for j, year in enumerate(YEARS):
        sub = tbl[tbl["year"] == year].set_index("model").reindex(MODELS)
        offset = (j - (n_years - 1) / 2) * bar_width
        bars = ax.bar(x + offset, sub["diff"], bar_width,
                      yerr=sub["se"], capsize=3,
                      color=year_colors[year], edgecolor="white",
                      linewidth=0.5, label=year, alpha=0.85)
        for i, (_, row) in enumerate(sub.iterrows()):
            p = row["p_value"]
            star = "*" if p < 0.05 else ""
            ypos = row["diff"] + row["se"] + 0.05 * max(abs(tbl["diff"].max()), 1)
            if row["diff"] < 0:
                ypos = row["diff"] - row["se"] - 0.05 * max(abs(tbl["diff"].min()), 1)
            ax.text(x[i] + offset, ypos,
                    f"p={p:.2f}{star}", ha="center", va="bottom" if row["diff"] >= 0 else "top",
                    fontsize=7)

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel(r"$\Delta$ rate ($\mathcal{T}_A - \mathcal{T}_B$) (pp)")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, "fig_5_4_effect_size_bars.pdf")


if __name__ == "__main__":
    daily = load_daily_rates()
    tbl = compute_welch_table(daily)

    print("Welch t-test results:")
    print(tbl.to_string(index=False))
    print()

    fig_ta_vs_tb_dots(daily)
    fig_effect_size_bars(tbl)

    tbl.to_csv(os.path.join(OUT_DIR, "proximity_welch_stats.csv"), index=False)
    print(f"  saved {os.path.join(OUT_DIR, 'proximity_welch_stats.csv')}")
    print("\nDone.")
