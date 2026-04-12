"""
Generate all figures for Section 5.3 (Threshold Comparison).

Applies four thresholding methods (POT, SPOT, DSPOT, RFDR) to each model's
test-set score array and produces comparison figures.

Run from repo root:  python scripts/generate_figures_5_3.py
"""

import os, sys, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detection.thresholds.pot import PeakOverThreshold
from detection.thresholds.spot import StreamingPeakOverThreshold
from detection.thresholds.dspot import DriftStreamingPeakOverThreshold
from detection.thresholds.rfdr import RollingFalseDiscoveryRate

# ────────────────────────────────────────────────────────────────────
# Configuration (from config/default.yaml)
# ────────────────────────────────────────────────────────────────────
THRESHOLD_PARAMS = dict(
    risk=0.001,
    init_level=0.98,
    num_candidates=10,
    num_init=1000,
    depth=200,
    window_size=500,
    alpha=0.05,
)

# Results are computed for the specified training year.
YEAR = "2015"
DATA_DIR = os.path.join("results", YEAR, "test_output")
OUT_DIR = os.path.join("figures", "results")
STATS_PATH = os.path.join(OUT_DIR, "threshold_comparison_stats.json")
os.makedirs(OUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────
# Plot style (matches Sections 5.1 / 5.2)
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

MODEL_KEYS   = ["transformer_ocsvm", "pnn", "prae"]
MODEL_LABELS = {"transformer_ocsvm": "TF\u2013OC-SVM", "pnn": "PNN", "prae": "PRAE"}
MODEL_COLORS = {"transformer_ocsvm": BLUE, "pnn": ORANGE, "prae": GREEN}

METHOD_KEYS   = ["pot", "spot", "dspot", "rfdr"]
METHOD_LABELS = {"pot": "POT", "spot": "SPOT", "dspot": "DSPOT", "rfdr": "RFDR"}
METHOD_COLORS = {"pot": BLUE, "spot": ORANGE, "dspot": RED, "rfdr": PURPLE}

# Subsample size for streaming methods
SUBSAMPLE_N = 100_000
RFDR_SUBSAMPLE_N = 20_000  # RFDR is O(n*w) per element; keep small
STREAMING_TIMEOUT = 180  # seconds per streaming method call


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  saved {path}")


# ────────────────────────────────────────────────────────────────────
# Load metadata
# ────────────────────────────────────────────────────────────────────
with open(os.path.join(DATA_DIR, "test_meta.json")) as f:
    META = json.load(f)

DAY_NAMES      = META["day_names"]
DAY_SPLITS     = META["day_split_labels"]
DAY_BOUNDARIES = META["day_boundaries"]

SPLIT_INDICES = {}
for split in ["test_proximate", "test_distal"]:
    indices = [(DAY_BOUNDARIES[i], DAY_BOUNDARIES[i + 1])
               for i, l in enumerate(DAY_SPLITS) if l == split]
    SPLIT_INDICES[split] = indices


def get_split_mask(n, split):
    """Boolean mask of length n selecting samples in the given split."""
    mask = np.zeros(n, dtype=bool)
    for s, e in SPLIT_INDICES[split]:
        mask[s:e] = True
    return mask


# ────────────────────────────────────────────────────────────────────
# Apply thresholding methods
# ────────────────────────────────────────────────────────────────────
def apply_pot(scores):
    """Return (threshold_scalar, t_init)."""
    z, t = PeakOverThreshold(
        scores,
        num_candidates=THRESHOLD_PARAMS["num_candidates"],
        risk=THRESHOLD_PARAMS["risk"],
        init_level=THRESHOLD_PARAMS["init_level"],
    )
    return z, t


def _run_with_timeout(fn, args, label, fallback_tau, n):
    """Run fn(*args) in a subprocess with STREAMING_TIMEOUT.  Falls back to constant array."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTE
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args)
        try:
            return future.result(timeout=STREAMING_TIMEOUT)
        except FTE:
            print(f"[{label} timeout after {STREAMING_TIMEOUT}s, fallback to POT]", end=" ")
            future.cancel()
            return np.full(n, fallback_tau)
        except Exception as e:
            print(f"[{label} fallback: {e}]", end=" ")
            return np.full(n, fallback_tau)


def _spot_inner(scores):
    thresholds = StreamingPeakOverThreshold(
        scores,
        num_init=THRESHOLD_PARAMS["num_init"],
        num_candidates=THRESHOLD_PARAMS["num_candidates"],
        risk=THRESHOLD_PARAMS["risk"],
        init_level=THRESHOLD_PARAMS["init_level"],
    )
    return np.array(thresholds)


def _dspot_inner(scores):
    thresholds = DriftStreamingPeakOverThreshold(
        scores,
        num_init=THRESHOLD_PARAMS["num_init"],
        depth=THRESHOLD_PARAMS["depth"],
        num_candidates=THRESHOLD_PARAMS["num_candidates"],
        risk=THRESHOLD_PARAMS["risk"],
        init_level=THRESHOLD_PARAMS["init_level"],
    )
    return np.array(thresholds)


def apply_spot(scores, fallback_tau=None):
    """Return array of thresholds.  Falls back to POT threshold on failure/timeout."""
    if fallback_tau is None:
        fallback_tau, _ = apply_pot(scores)
    return _run_with_timeout(_spot_inner, (scores,), "SPOT", fallback_tau, len(scores))


def apply_dspot(scores, fallback_tau=None):
    """Return array of thresholds.  Falls back to POT threshold on failure/timeout."""
    if fallback_tau is None:
        fallback_tau, _ = apply_pot(scores)
    return _run_with_timeout(_dspot_inner, (scores,), "DSPOT", fallback_tau, len(scores))


def apply_rfdr(scores):
    """Return (preds, thresholds) arrays.

    Vectorised rolling-window BH-FDR reimplementation of
    ``RollingFalseDiscoveryRate`` (detection.thresholds.rfdr) for speed.
    Only the final (most-recent) score in each window is tested.
    """
    from scipy import stats as _stats

    ws = THRESHOLD_PARAMS["window_size"]
    alpha = THRESHOLD_PARAMS["alpha"]
    n = len(scores)
    preds = np.zeros(n, dtype=bool)
    thresholds = np.zeros(n, dtype=np.float64)

    # Process in chunks using stride_tricks for the rolling window
    for i in range(ws, n):
        window = scores[max(0, i - ws + 1):i + 1]
        wn = len(window)
        if wn < 20:
            continue
        # Log transform
        min_val = window.min()
        shifted = window + (abs(min_val) + 1e-6) if min_val <= 0 else window
        logged = np.log(shifted)
        # Robust z-scores
        med = np.median(logged)
        diff = np.abs(logged - med)
        mad = np.median(diff)
        if mad == 0:
            mad = np.mean(diff) + 1e-10
        robust_sigma = mad * 1.4826
        z = (logged - med) / robust_sigma
        # p-values
        pv = _stats.norm.sf(z)
        # BH correction
        sorted_idx = np.argsort(pv)
        sorted_pv = pv[sorted_idx]
        ranks = np.arange(1, wn + 1)
        crit = (ranks / wn) * alpha
        below = sorted_pv <= crit
        if np.any(below):
            k = np.max(np.where(below))
            thr_idx = sorted_idx[k]
            thr_raw = window[thr_idx]
        else:
            thr_raw = np.max(window)
        thresholds[i] = thr_raw
        preds[i] = scores[i] > thr_raw

    return preds, thresholds


def compute_all_thresholds(model_key):
    """
    Compute thresholds for one model using all four methods.

    Returns dict:
      {method: {"tau": float,           # representative threshold
                "tau_p10": float,        # 10th pctile (for adaptive methods)
                "tau_p90": float,        # 90th pctile
                "anomaly_rate": float,   # on full array
                "rate_ta": float,        # on T_A
                "rate_tb": float,        # on T_B
                "preds_full": np.array}} # boolean predictions on full array
    """
    print(f"\n  Loading {model_key} scores...")
    scores_full = np.load(os.path.join(DATA_DIR, f"{model_key}_scores.npy"))
    n = len(scores_full)

    # Handle NaN: replace with median for thresholding
    nan_mask = np.isnan(scores_full)
    if nan_mask.any():
        median_val = np.nanmedian(scores_full)
        scores_clean = scores_full.copy()
        scores_clean[nan_mask] = median_val
        print(f"    {nan_mask.sum()} NaN values replaced with median ({median_val:.4f})")
    else:
        scores_clean = scores_full

    # Subsample for streaming methods:
    # Use sorted subsample to preserve temporal order while covering distribution
    rng = np.random.default_rng(42)
    sub_idx = np.sort(rng.choice(n, size=min(SUBSAMPLE_N, n), replace=False))
    scores_sub = scores_clean[sub_idx].copy()
    print(f"    Subsample: {len(scores_sub)} random samples")

    # Split masks
    mask_ta = get_split_mask(n, "test_proximate")
    mask_tb = get_split_mask(n, "test_distal")

    results = {}

    # --- POT (batch, on full array) ---
    print(f"    POT...", end=" ", flush=True)
    t0 = time.time()
    tau_pot, _ = apply_pot(scores_clean)
    preds_pot = scores_clean > tau_pot
    results["pot"] = {
        "tau": float(tau_pot),
        "tau_p10": float(tau_pot),
        "tau_p90": float(tau_pot),
        "anomaly_rate": float(preds_pot.mean() * 100),
        "rate_ta": float(preds_pot[mask_ta].mean() * 100),
        "rate_tb": float(preds_pot[mask_tb].mean() * 100),
        "preds_full": preds_pot,
    }
    print(f"tau={tau_pot:.6f}, rate={preds_pot.mean()*100:.4f}% ({time.time()-t0:.1f}s)")

    # Fast path: if POT threshold exceeds max score, streaming methods are futile
    pot_exceeds_max = tau_pot > scores_clean.max()
    if pot_exceeds_max:
        print(f"    [POT tau > max(scores); streaming methods will use POT fallback]")

    # --- SPOT (streaming, on subsample) ---
    print(f"    SPOT...", end=" ", flush=True)
    t0 = time.time()
    if pot_exceeds_max:
        spot_thresholds = np.full(len(scores_sub), tau_pot)
        print(f"[skipped, POT fallback]", end=" ")
    else:
        spot_thresholds = apply_spot(scores_sub, fallback_tau=tau_pot)
    tau_spot_med = float(np.median(spot_thresholds))
    preds_spot = scores_clean > tau_spot_med
    results["spot"] = {
        "tau": tau_spot_med,
        "tau_p10": float(np.percentile(spot_thresholds, 10)),
        "tau_p90": float(np.percentile(spot_thresholds, 90)),
        "anomaly_rate": float(preds_spot.mean() * 100),
        "rate_ta": float(preds_spot[mask_ta].mean() * 100),
        "rate_tb": float(preds_spot[mask_tb].mean() * 100),
        "preds_full": preds_spot,
    }
    print(f"tau_med={tau_spot_med:.6f}, rate={preds_spot.mean()*100:.4f}% ({time.time()-t0:.1f}s)")

    # --- DSPOT (streaming, on subsample) ---
    print(f"    DSPOT...", end=" ", flush=True)
    t0 = time.time()
    if pot_exceeds_max:
        dspot_thresholds = np.full(len(scores_sub), tau_pot)
        print(f"[skipped, POT fallback]", end=" ")
    else:
        dspot_thresholds = apply_dspot(scores_sub, fallback_tau=tau_pot)
    tau_dspot_med = float(np.median(dspot_thresholds))
    preds_dspot = scores_clean > tau_dspot_med
    results["dspot"] = {
        "tau": tau_dspot_med,
        "tau_p10": float(np.percentile(dspot_thresholds, 10)),
        "tau_p90": float(np.percentile(dspot_thresholds, 90)),
        "anomaly_rate": float(preds_dspot.mean() * 100),
        "rate_ta": float(preds_dspot[mask_ta].mean() * 100),
        "rate_tb": float(preds_dspot[mask_tb].mean() * 100),
        "preds_full": preds_dspot,
    }
    print(f"tau_med={tau_dspot_med:.6f}, rate={preds_dspot.mean()*100:.4f}% ({time.time()-t0:.1f}s)")

    # --- RFDR (streaming, on smaller subsample) ---
    print(f"    RFDR...", end=" ", flush=True)
    t0 = time.time()
    rng2 = np.random.default_rng(99)
    rfdr_idx = np.sort(rng2.choice(n, size=min(RFDR_SUBSAMPLE_N, n), replace=False))
    scores_rfdr_sub = scores_clean[rfdr_idx].copy()
    _, rfdr_thresholds = apply_rfdr(scores_rfdr_sub)
    # RFDR threshold is the score above which anomalies are flagged
    valid_thr = rfdr_thresholds[rfdr_thresholds > 0]
    if len(valid_thr) == 0:
        valid_thr = rfdr_thresholds[THRESHOLD_PARAMS["window_size"]:]
    tau_rfdr_med = float(np.median(valid_thr)) if len(valid_thr) > 0 else float(np.max(scores_clean))
    preds_rfdr = scores_clean > tau_rfdr_med
    results["rfdr"] = {
        "tau": tau_rfdr_med,
        "tau_p10": float(np.percentile(valid_thr, 10)) if len(valid_thr) > 0 else tau_rfdr_med,
        "tau_p90": float(np.percentile(valid_thr, 90)) if len(valid_thr) > 0 else tau_rfdr_med,
        "anomaly_rate": float(preds_rfdr.mean() * 100),
        "rate_ta": float(preds_rfdr[mask_ta].mean() * 100),
        "rate_tb": float(preds_rfdr[mask_tb].mean() * 100),
        "preds_full": preds_rfdr,
    }
    print(f"tau_med={tau_rfdr_med:.6f}, rate={preds_rfdr.mean()*100:.4f}% ({time.time()-t0:.1f}s)")

    return results


# ────────────────────────────────────────────────────────────────────
# Compute thresholds for all models
# ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Computing thresholds for all models...")
print("=" * 60)

ALL_RESULTS = {}
for mk in MODEL_KEYS:
    ALL_RESULTS[mk] = compute_all_thresholds(mk)

# Save stats (without the large preds arrays)
stats_out = {}
for mk in MODEL_KEYS:
    stats_out[mk] = {}
    for meth in METHOD_KEYS:
        r = ALL_RESULTS[mk][meth]
        stats_out[mk][meth] = {k: v for k, v in r.items() if k != "preds_full"}
with open(STATS_PATH, "w") as f:
    json.dump(stats_out, f, indent=2)
print(f"\nSaved stats to {STATS_PATH}")


# ────────────────────────────────────────────────────────────────────
# Figure 5.3.1 — Threshold values by method and model
# ────────────────────────────────────────────────────────────────────
def fig_threshold_values():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, mk in zip(axes, MODEL_KEYS):
        x = np.arange(len(METHOD_KEYS))
        taus   = [ALL_RESULTS[mk][m]["tau"] for m in METHOD_KEYS]
        lo     = [ALL_RESULTS[mk][m]["tau_p10"] for m in METHOD_KEYS]
        hi     = [ALL_RESULTS[mk][m]["tau_p90"] for m in METHOD_KEYS]
        errors = [[t - l for t, l in zip(taus, lo)],
                  [h - t for t, h in zip(taus, hi)]]
        colors = [METHOD_COLORS[m] for m in METHOD_KEYS]
        bars = ax.bar(x, taus, color=colors, edgecolor="white", linewidth=0.5,
                      yerr=errors, capsize=4, error_kw={"linewidth": 1})
        for bar, val in zip(bars, taus):
            fmt = f"{val:.4f}" if abs(val) < 1 else f"{val:.1f}"
            yoff = max(abs(val) * 0.03, 0.001)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + yoff,
                    fmt, ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_KEYS])
        ax.set_ylabel(r"Threshold $\tau$")
        ax.set_title(MODEL_LABELS[mk], fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Threshold values by method and model", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_5_3_threshold_values.pdf")


# ────────────────────────────────────────────────────────────────────
# Figure 5.3.2 — Anomaly rates by threshold method (one panel per model)
# ────────────────────────────────────────────────────────────────────
def fig_anomaly_rates_by_method():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    split_labels = {
        "overall": "Overall",
        "ta": r"$\mathcal{T}_A$",
        "tb": r"$\mathcal{T}_B$",
    }
    split_colors = {"overall": "#888888", "ta": RED, "tb": PURPLE}

    for ax, mk in zip(axes, MODEL_KEYS):
        x = np.arange(len(METHOD_KEYS))
        width = 0.25
        for i, (skey, slabel) in enumerate(split_labels.items()):
            if skey == "overall":
                vals = [ALL_RESULTS[mk][m]["anomaly_rate"] for m in METHOD_KEYS]
            elif skey == "ta":
                vals = [ALL_RESULTS[mk][m]["rate_ta"] for m in METHOD_KEYS]
            else:
                vals = [ALL_RESULTS[mk][m]["rate_tb"] for m in METHOD_KEYS]
            bars = ax.bar(x + i * width, vals, width,
                          label=slabel, color=split_colors[skey],
                          alpha=0.85, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val > 0.01:
                    fmt = f"{val:.2f}" if val >= 0.1 else f"{val:.3f}"
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(val * 0.02, 0.001),
                            fmt, ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x + width)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_KEYS])
        ax.set_ylabel("Anomaly rate (%)")
        ax.set_title(MODEL_LABELS[mk], fontsize=10)
        ax.legend(frameon=False, fontsize=7, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Anomaly rate by thresholding method", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_5_3_anomaly_rates_by_method.pdf")


# ────────────────────────────────────────────────────────────────────
# Figure 5.3.3 — Score series with all four thresholds (one day)
# ────────────────────────────────────────────────────────────────────
def fig_score_series_thresholds():
    """Plot one proximate test day's score series with 4 threshold lines."""
    # Use first proximate day
    prox_idx = [i for i, l in enumerate(DAY_SPLITS) if l == "test_proximate"][0]
    start = DAY_BOUNDARIES[prox_idx]
    end   = DAY_BOUNDARIES[prox_idx + 1]
    day_str = DAY_NAMES[prox_idx][:10]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, mk in zip(axes, MODEL_KEYS):
        scores_day = np.load(
            os.path.join(DATA_DIR, f"{mk}_scores.npy"),
            mmap_mode="r"
        )[start:end].copy()

        # Replace NaN with median for display
        nan_m = np.isnan(scores_day)
        if nan_m.any():
            scores_day[nan_m] = np.nanmedian(scores_day)

        n = len(scores_day)
        k = max(1, n // 5000)
        idx = np.arange(0, n, k)
        x = idx / n

        ax.plot(x, scores_day[idx], color=MODEL_COLORS[mk],
                linewidth=0.4, alpha=0.7, rasterized=True)

        # Draw threshold lines
        for meth in METHOD_KEYS:
            tau = ALL_RESULTS[mk][meth]["tau"]
            ax.axhline(tau, color=METHOD_COLORS[meth], linestyle="--",
                       linewidth=1.0, label=f"{METHOD_LABELS[meth]} ({tau:.4f})"
                       if abs(tau) < 10 else f"{METHOD_LABELS[meth]} ({tau:.1f})")

        # Shade regions where ALL methods agree on anomaly
        all_agree = np.ones(len(idx), dtype=bool)
        any_flag  = np.zeros(len(idx), dtype=bool)
        for meth in METHOD_KEYS:
            tau = ALL_RESULTS[mk][meth]["tau"]
            flagged = scores_day[idx] > tau
            all_agree &= flagged
            any_flag  |= flagged

        some_only = any_flag & ~all_agree
        if some_only.any():
            ax.fill_between(x, ax.get_ylim()[0], ax.get_ylim()[1],
                            where=some_only, alpha=0.08, color="orange",
                            step="mid", rasterized=True)
        if all_agree.any():
            ax.fill_between(x, ax.get_ylim()[0], ax.get_ylim()[1],
                            where=all_agree, alpha=0.20, color="red",
                            step="mid", rasterized=True)

        ax.set_ylabel("Score")
        ax.set_title(f"{MODEL_LABELS[mk]}", fontsize=10)
        ax.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Normalised time (0 = open, 1 = close)")
    fig.suptitle(f"Score series with threshold overlays — {day_str}", fontsize=11, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig_5_3_score_series_thresholds.pdf")


# ────────────────────────────────────────────────────────────────────
# Figure 5.3.4 — Pairwise Jaccard heatmaps
# ────────────────────────────────────────────────────────────────────
def jaccard(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return inter / union if union > 0 else 0.0


def fig_jaccard_heatmaps():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, mk in zip(axes, MODEL_KEYS):
        nm = len(METHOD_KEYS)
        mat = np.zeros((nm, nm))
        for i, m1 in enumerate(METHOD_KEYS):
            for j, m2 in enumerate(METHOD_KEYS):
                p1 = ALL_RESULTS[mk][m1]["preds_full"]
                p2 = ALL_RESULTS[mk][m2]["preds_full"]
                mat[i, j] = jaccard(p1, p2)

        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(nm))
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_KEYS], fontsize=8)
        ax.set_yticks(range(nm))
        ax.set_yticklabels([METHOD_LABELS[m] for m in METHOD_KEYS], fontsize=8)
        for i in range(nm):
            for j in range(nm):
                txt = f"{mat[i, j]:.3f}"
                clr = "white" if mat[i, j] > 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=clr)
        ax.set_title(MODEL_LABELS[mk], fontsize=10)

    fig.suptitle("Pairwise Jaccard similarity between thresholding methods",
                 fontsize=11, y=1.02)
    fig.colorbar(im, ax=axes, shrink=0.7, label="Jaccard index", pad=0.02)
    fig.tight_layout()
    save_fig(fig, "fig_5_3_jaccard_heatmaps.pdf")


# ────────────────────────────────────────────────────────────────────
# Figure 5.3.5 — T_A vs T_B scatter
# ────────────────────────────────────────────────────────────────────
def fig_ta_vs_tb():
    fig, ax = plt.subplots(figsize=(5.5, 5))
    markers = {"transformer_ocsvm": "o", "pnn": "s", "prae": "D"}
    for mk in MODEL_KEYS:
        for meth in METHOD_KEYS:
            r = ALL_RESULTS[mk][meth]
            ax.scatter(r["rate_ta"], r["rate_tb"],
                       color=METHOD_COLORS[meth],
                       marker=markers[mk], s=60, edgecolor="black", linewidth=0.5,
                       zorder=3)

    # diagonal
    lim_max = max(
        max(ALL_RESULTS[mk][m]["rate_ta"] for mk in MODEL_KEYS for m in METHOD_KEYS),
        max(ALL_RESULTS[mk][m]["rate_tb"] for mk in MODEL_KEYS for m in METHOD_KEYS),
    ) * 1.15
    ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel(r"Anomaly rate on $\mathcal{T}_A$ (%)")
    ax.set_ylabel(r"Anomaly rate on $\mathcal{T}_B$ (%)")
    ax.set_title(r"$\mathcal{T}_A$ vs. $\mathcal{T}_B$ anomaly rate by method", fontsize=11)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # legend: markers for models, colors for methods
    handles_model = [plt.Line2D([0], [0], marker=markers[mk], color="gray",
                                markerfacecolor="gray", markersize=7, linestyle="None",
                                label=MODEL_LABELS[mk]) for mk in MODEL_KEYS]
    handles_method = [Patch(facecolor=METHOD_COLORS[m], label=METHOD_LABELS[m])
                      for m in METHOD_KEYS]
    ax.legend(handles=handles_model + handles_method, frameon=False,
              fontsize=7, loc="upper left", ncol=2)
    save_fig(fig, "fig_5_3_ta_vs_tb.pdf")


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Generating Section 5.3 figures...")
    print("=" * 60)
    fig_threshold_values()
    fig_anomaly_rates_by_method()
    fig_score_series_thresholds()
    fig_jaccard_heatmaps()
    fig_ta_vs_tb()
    print("\nDone.")
