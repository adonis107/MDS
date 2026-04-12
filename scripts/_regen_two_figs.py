"""Quick standalone script to regenerate only fig_5_3_ta_vs_tb.pdf
and fig_5_8_price_path.pdf from existing data, without recomputing
thresholds or running the full pipeline."""

import os, sys, json, warnings, gc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

OUT_DIR = os.path.join("figures", "results")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
BLUE   = "#4878CF"
ORANGE = "#E8884A"
RED    = "#C44E52"
GREEN  = "#6AB187"
PURPLE = "#9B59B6"
GREY   = "#999999"

MODEL_KEYS   = ["transformer_ocsvm", "pnn", "prae"]
MODEL_LABELS = {"transformer_ocsvm": "TF\u2013OC-SVM", "pnn": "PNN", "prae": "PRAE"}
METHOD_KEYS   = ["pot", "spot", "dspot", "rfdr"]
METHOD_LABELS = {"pot": "POT", "spot": "SPOT", "dspot": "DSPOT", "rfdr": "RFDR"}
METHOD_COLORS = {"pot": BLUE, "spot": ORANGE, "dspot": RED, "rfdr": PURPLE}
MODEL_COLORS = {"transformer_ocsvm": BLUE, "pnn": ORANGE, "prae": GREEN, "consensus": RED}


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  saved {path}")


# ── Figure 1: T_A vs T_B with inset zoom ──────────────────────────
def regen_ta_vs_tb():
    stats_path = os.path.join("figures", "results", "threshold_comparison_stats.json")
    with open(stats_path) as f:
        stats = json.load(f)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    markers = {"transformer_ocsvm": "o", "pnn": "s", "prae": "D"}

    all_ta, all_tb = [], []
    for mk in MODEL_KEYS:
        for meth in METHOD_KEYS:
            r = stats[mk][meth]
            all_ta.append(r["rate_ta"])
            all_tb.append(r["rate_tb"])
            ax.scatter(r["rate_ta"], r["rate_tb"],
                       color=METHOD_COLORS[meth],
                       marker=markers[mk], s=60, edgecolor="black", linewidth=0.5,
                       zorder=3)

    lim_max = max(max(all_ta), max(all_tb)) * 1.15
    ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel(r"Anomaly rate on $\mathcal{T}_A$ (%)")
    ax.set_ylabel(r"Anomaly rate on $\mathcal{T}_B$ (%)")
    ax.set_title(r"$\mathcal{T}_A$ vs. $\mathcal{T}_B$ anomaly rate by method", fontsize=11)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Inset zoom on low-rate region ──
    low_ta = [v for v in all_ta if v <= 5]
    low_tb = [v for v in all_tb if v <= 5]
    if low_ta and low_tb:
        zoom_max = max(max(low_ta), max(low_tb)) * 1.25
        axins = inset_axes(ax, width="45%", height="45%", loc="center right",
                           borderpad=1.5)
        for mk in MODEL_KEYS:
            for meth in METHOD_KEYS:
                r = stats[mk][meth]
                axins.scatter(r["rate_ta"], r["rate_tb"],
                              color=METHOD_COLORS[meth],
                              marker=markers[mk], s=40, edgecolor="black",
                              linewidth=0.4, zorder=3)
        axins.plot([0, zoom_max], [0, zoom_max], "k--", linewidth=0.6, alpha=0.4)
        axins.set_xlim(0, zoom_max)
        axins.set_ylim(0, zoom_max)
        axins.set_aspect("equal")
        axins.tick_params(labelsize=7)
        axins.set_title("Zoom (rates \u2264 5%)", fontsize=8)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5",
                   linewidth=0.8, linestyle="--")

    handles_model = [plt.Line2D([0], [0], marker=markers[mk], color="gray",
                                markerfacecolor="gray", markersize=7, linestyle="None",
                                label=MODEL_LABELS[mk]) for mk in MODEL_KEYS]
    handles_method = [Patch(facecolor=METHOD_COLORS[m], label=METHOD_LABELS[m])
                      for m in METHOD_KEYS]
    ax.legend(handles=handles_model + handles_method, frameon=False,
              fontsize=7, loc="upper left", ncol=2)
    save_fig(fig, "fig_5_3_ta_vs_tb.pdf")


# ── Figure 2: Price path ──────────────────────────────────────────
SEQ_LENGTH = 25
YEARS = ["2015", "2017"]
MODEL_TYPES = ["transformer_ocsvm", "pnn", "prae"]
DATA_DIR_LOB = "data/processed/TOTF.PA-book"
EVENT_STUDY_HALF = 250
N_ES_MAX = 5000
N_BOOTSTRAP = 500


def analyze_price_path(year):
    print(f"\n  Year {year}")
    with open(f"results/{year}/test_output/test_meta.json") as f:
        meta = json.load(f)
    bounds = meta["day_boundaries"]
    day_names = meta["day_names"]
    test_days = [i for i, l in enumerate(meta["day_split_labels"]) if "test" in l]
    test_start = bounds[test_days[0]]
    test_end = bounds[test_days[-1] + 1]
    n_test = test_end - test_start

    preds = {}
    for mt in MODEL_TYPES:
        p = np.load(f"results/{year}/test_output/{mt}_preds.npy")[test_start:test_end]
        preds[mt] = p.astype(bool)
    n_models = sum(preds[mt].astype(np.int8) for mt in MODEL_TYPES)
    any_model = n_models >= 1

    print(f"    Test windows: {n_test:,}, Any-model flagged: {any_model.sum():,}")

    rng = np.random.default_rng(42)
    flagged_idx = np.where(any_model)[0]
    if len(flagged_idx) > N_ES_MAX:
        es_idx = np.sort(rng.choice(flagged_idx, N_ES_MAX, replace=False))
    else:
        es_idx = flagged_idx.copy()

    nf_pool = np.where(~any_model)[0]
    nf_n = min(len(es_idx), len(nf_pool))
    nf_sample = np.sort(rng.choice(nf_pool, nf_n, replace=False)) if nf_n > 0 \
        else np.array([], dtype=int)

    HALF = EVENT_STUDY_HALF
    PL = 2 * HALF + 1

    day_data = []
    for d in test_days:
        dn = day_names[d]
        fp = os.path.join(DATA_DIR_LOB, dn)
        if not os.path.exists(fp):
            continue
        df = pd.read_parquet(fp, columns=["bid-price-1", "ask-price-1"])
        mid = ((df["bid-price-1"] + df["ask-price-1"]) / 2).values.astype(np.float64)
        n_seqs = bounds[d + 1] - bounds[d]
        local_start = bounds[d] - test_start
        day_data.append({"mid": mid, "n_seqs": n_seqs, "local_start": local_start})
        del df

    def extract_paths(indices):
        paths = []
        for dd in day_data:
            mid = dd["mid"]
            ls = dd["local_start"]
            le = ls + dd["n_seqs"]
            sel = indices[(indices >= ls) & (indices < le)] - ls
            for j in sel:
                a = j + SEQ_LENGTH - 1
                s, e = a - HALF, a + HALF + 1
                if s >= 0 and e <= len(mid) and mid[a] > 0:
                    paths.append(np.log(mid[s:e] / mid[a]) * 1e4)
        return np.array(paths) if paths else np.empty((0, PL))

    print(f"    Extracting paths ({len(es_idx)} flagged, {len(nf_sample)} nf)...")
    fp = extract_paths(es_idx)
    nfp = extract_paths(nf_sample)
    print(f"    Got {len(fp)} flagged paths, {len(nfp)} non-flagged paths")

    # Check for issues
    if len(fp) > 0:
        n_nan = np.isnan(fp).sum()
        n_inf = np.isinf(fp).sum()
        if n_nan > 0 or n_inf > 0:
            print(f"    WARNING: {n_nan} NaN, {n_inf} Inf in flagged paths")
            # Replace non-finite with NaN for nanmean
            fp = np.where(np.isfinite(fp), fp, np.nan)

    if len(nfp) > 0:
        nfp = np.where(np.isfinite(nfp), nfp, np.nan)

    if len(fp) >= 2:
        mean_f = np.nanmean(fp, axis=0)
        bm = np.empty((N_BOOTSTRAP, PL))
        for b in range(N_BOOTSTRAP):
            bm[b] = np.nanmean(fp[rng.choice(len(fp), len(fp), replace=True)], axis=0)
        ci_lo = np.nanpercentile(bm, 2.5, axis=0)
        ci_hi = np.nanpercentile(bm, 97.5, axis=0)
    else:
        mean_f = ci_lo = ci_hi = np.full(PL, np.nan)

    mean_nf = np.nanmean(nfp, axis=0) if len(nfp) >= 2 else np.full(PL, np.nan)

    return dict(year=year, mean_flagged_path=mean_f, ci_lo=ci_lo, ci_hi=ci_hi,
                mean_nonflagged_path=mean_nf, n_flagged_paths=len(fp),
                n_nonflagged_paths=len(nfp))


def regen_price_path():
    results = []
    for year in YEARS:
        try:
            results.append(analyze_price_path(year))
        except Exception as e:
            print(f"  ERROR for {year}: {e}")

    if not results:
        print("  No results to plot!")
        return

    n_years = len(results)
    fig, axes = plt.subplots(n_years, 1, figsize=(8, 4 * n_years), squeeze=False)
    t = np.arange(-EVENT_STUDY_HALF, EVENT_STUDY_HALF + 1)

    for row, res in enumerate(results):
        ax = axes[row, 0]
        year = res["year"]
        mf = res["mean_flagged_path"]
        mnf = res["mean_nonflagged_path"]

        finite_mnf = np.isfinite(mnf)
        finite_mf = np.isfinite(mf)
        if np.any(finite_mnf):
            ax.plot(t, np.where(finite_mnf, mnf, np.nan),
                    color=GREY, lw=1, label="Non-flagged", zorder=2)
        if np.any(finite_mf):
            mf_safe = np.where(finite_mf, mf, np.nan)
            ax.plot(t, mf_safe, color=BLUE, lw=1.5,
                    label=r"Flagged ($\geq$1 model)", zorder=3)
            ci_lo = np.where(finite_mf, res["ci_lo"], np.nan)
            ci_hi = np.where(finite_mf, res["ci_hi"], np.nan)
            ax.fill_between(t, ci_lo, ci_hi, color=BLUE, alpha=0.2, zorder=2)

        ax.axvline(0, color="black", ls="--", lw=0.8, alpha=0.5)
        ax.axhline(0, color="black", lw=0.5, alpha=0.3)
        ax.set_xlabel("Events relative to flagged window")
        ax.set_ylabel("Cumulative log-return (bps)")
        ax.set_title(f"{year}: mean mid-price path around flagged windows "
                     f"(n = {res['n_flagged_paths']:,})")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "fig_5_8_price_path.pdf")


if __name__ == "__main__":
    print("=" * 60)
    print("Regenerating fig_5_3_ta_vs_tb.pdf ...")
    print("=" * 60)
    regen_ta_vs_tb()

    print("\n" + "=" * 60)
    print("Regenerating fig_5_8_price_path.pdf ...")
    print("=" * 60)
    regen_price_path()

    print("\nDone.")
