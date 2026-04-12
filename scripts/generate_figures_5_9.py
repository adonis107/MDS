"""
Cross-year robustness analysis for Section 5.9.

Compares anomaly rates, score distributions, and detection overlap
(Jaccard similarity) when models trained on one year are applied to
data from the other year, without refitting scalers or thresholds.

Run from repo root:  python scripts/generate_figures_5_9.py
"""

import os, json, warnings, gc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ── Constants ───────────────────────────────────────────────────────
OUT_DIR = os.path.join("figures", "results")
os.makedirs(OUT_DIR, exist_ok=True)

YEARS = ["2015", "2017"]
MODEL_TYPES = ["transformer_ocsvm", "pnn", "prae"]
MODEL_LABELS = {
    "transformer_ocsvm": "TF-OCSVM",
    "pnn": "PNN",
    "prae": "PRAE",
}

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
BLUE, ORANGE, GREEN, GREY = "#4878CF", "#E8884A", "#6AB187", "#999999"
MODEL_COLORS = {"transformer_ocsvm": BLUE, "pnn": ORANGE, "prae": GREEN}
YEAR_COLORS = {"2015": BLUE, "2017": ORANGE}

KDE_SUBSAMPLE = 50_000


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  saved {path}")


# ── Helpers ─────────────────────────────────────────────────────────

def load_meta(year):
    with open(os.path.join("results", year, "test_output", "test_meta.json")) as f:
        return json.load(f)


def load_array(year, filename):
    return np.load(
        os.path.join("results", year, "test_output", filename),
        mmap_mode="r",
    )


def day_bounds_dict(meta):
    """Return {day_name: (start_idx, end_idx)}."""
    bounds = meta["day_boundaries"]
    total = meta["total_samples"]
    out = {}
    for i, name in enumerate(meta["day_names"]):
        s = bounds[i]
        e = bounds[i + 1] if i + 1 < len(bounds) else total
        out[name] = (s, e)
    return out


def split_slice(meta, labels=("test_proximate", "test_distal")):
    """(start, end) for contiguous days matching *labels*."""
    bounds = meta["day_boundaries"]
    total = meta["total_samples"]
    idxs = [i for i, l in enumerate(meta["day_split_labels"]) if l in labels]
    if not idxs:
        return None
    return bounds[idxs[0]], (bounds[idxs[-1] + 1] if idxs[-1] + 1 < len(bounds) else total)


def cross_year_slice(meta, train_year):
    """(start, end) for all days from the *other* year."""
    cross = "2017" if train_year == "2015" else "2015"
    bounds = meta["day_boundaries"]
    total = meta["total_samples"]
    idxs = [i for i, n in enumerate(meta["day_names"]) if cross in n]
    if not idxs:
        return None
    return bounds[idxs[0]], (bounds[idxs[-1] + 1] if idxs[-1] + 1 < len(bounds) else total)


def inyear_test_day_names(meta):
    return [n for n, l in zip(meta["day_names"], meta["day_split_labels"])
            if l in ("test_proximate", "test_distal")]


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("Loading metadata …")
    metas = {y: load_meta(y) for y in YEARS}
    dbounds = {y: day_bounds_dict(metas[y]) for y in YEARS}

    # ================================================================
    # 1.  Anomaly rates  (in-year test vs full cross-year)
    # ================================================================
    print("\n── Anomaly rates ──")
    rates = {}                      # rates[train][scope][model]
    for ty in YEARS:
        rates[ty] = {"in_year": {}, "cross_year": {}}
        iy = split_slice(metas[ty])
        cx = cross_year_slice(metas[ty], ty)
        for mt in MODEL_TYPES:
            preds = load_array(ty, f"{mt}_preds.npy")
            rates[ty]["in_year"][mt] = float(np.asarray(preds[iy[0]:iy[1]]).mean()) * 100
            rates[ty]["cross_year"][mt] = float(np.asarray(preds[cx[0]:cx[1]]).mean()) * 100

    for ty in YEARS:
        cy = "2017" if ty == "2015" else "2015"
        for mt in MODEL_TYPES:
            print(f"  {ty}-trained {MODEL_LABELS[mt]:>8s}: "
                  f"in-year = {rates[ty]['in_year'][mt]:7.3f}%   "
                  f"cross ({cy}) = {rates[ty]['cross_year'][mt]:7.3f}%")

    # ── Figure 1: grouped-bar anomaly-rate comparison ───────────────
    print("\nFigure 1 …")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    bw = 0.30
    x = np.arange(len(MODEL_TYPES))

    for pi, target_year in enumerate(YEARS):
        ax = axes[pi]
        other = "2017" if target_year == "2015" else "2015"

        iy_vals = [rates[target_year]["in_year"][mt] for mt in MODEL_TYPES]
        cx_vals = [rates[other]["cross_year"][mt] for mt in MODEL_TYPES]

        bars1 = ax.bar(x - bw / 2, iy_vals, bw,
                       label=f"Trained {target_year} (in-year)",
                       color=[MODEL_COLORS[mt] for mt in MODEL_TYPES],
                       alpha=0.9, edgecolor="black", linewidth=0.5)
        bars2 = ax.bar(x + bw / 2, cx_vals, bw,
                       label=f"Trained {other} (cross-year)",
                       color=[MODEL_COLORS[mt] for mt in MODEL_TYPES],
                       alpha=0.4, edgecolor="black", linewidth=0.5, hatch="//")

        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            if h >= 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2, h,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[mt] for mt in MODEL_TYPES])
        ax.set_ylabel("Anomaly rate (%)")
        ax.set_title(f"Target: {target_year} data")
        ax.legend(fontsize=7)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    save_fig(fig, "fig_5_9_anomaly_rate.pdf")

    # ================================================================
    # 2.  Score-distribution shift  (KDE, 2 × 3 grid)
    # ================================================================
    print("\nFigure 2 …")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    rng = np.random.default_rng(42)

    for row, ty in enumerate(YEARS):
        cy = "2017" if ty == "2015" else "2015"
        iy = split_slice(metas[ty])
        cx = cross_year_slice(metas[ty], ty)

        for col, mt in enumerate(MODEL_TYPES):
            ax = axes[row, col]
            scores = load_array(ty, f"{mt}_scores.npy")

            iy_s = np.asarray(scores[iy[0]:iy[1]], dtype=np.float64)
            cx_s = np.asarray(scores[cx[0]:cx[1]], dtype=np.float64)

            # Remove non-finite values
            iy_s = iy_s[np.isfinite(iy_s)]
            cx_s = cx_s[np.isfinite(cx_s)]

            iy_sub = rng.choice(iy_s, min(KDE_SUBSAMPLE, len(iy_s)), replace=False)
            cx_sub = rng.choice(cx_s, min(KDE_SUBSAMPLE, len(cx_s)), replace=False)

            combined = np.concatenate([iy_sub, cx_sub])
            lo, hi = np.percentile(combined, [1, 99])
            grid = np.linspace(lo, hi, 500)

            kde_iy = gaussian_kde(iy_sub)
            kde_cx = gaussian_kde(cx_sub)

            ax.plot(grid, kde_iy(grid), color=BLUE, lw=1.5,
                    label=f"In-year ({ty})")
            ax.fill_between(grid, kde_iy(grid), alpha=0.2, color=BLUE)
            ax.plot(grid, kde_cx(grid), color=ORANGE, lw=1.5, ls="--",
                    label=f"Cross-year ({cy})")
            ax.fill_between(grid, kde_cx(grid), alpha=0.15, color=ORANGE)

            if row == 0:
                ax.set_title(MODEL_LABELS[mt])
            if col == 0:
                ax.set_ylabel(f"Trained {ty}\nDensity")
            ax.legend(fontsize=7)
            ax.tick_params(axis="y", labelleft=False)

    fig.tight_layout()
    save_fig(fig, "fig_5_9_score_distribution.pdf")

    del scores, iy_s, cx_s
    gc.collect()

    # ================================================================
    # 3.  Detection overlap  (Jaccard on shared test days)
    # ================================================================
    print("\nFigure 3 …")
    jaccard_rows = []

    for target_year in YEARS:
        other = "2017" if target_year == "2015" else "2015"
        test_days = inyear_test_day_names(metas[target_year])

        for mt in MODEL_TYPES:
            iy_preds = load_array(target_year, f"{mt}_preds.npy")
            cx_preds = load_array(other, f"{mt}_preds.npy")

            iy_parts, cx_parts = [], []
            for dn in test_days:
                if dn in dbounds[target_year] and dn in dbounds[other]:
                    is_, ie = dbounds[target_year][dn]
                    cs, ce = dbounds[other][dn]
                    iy_d = np.asarray(iy_preds[is_:ie])
                    cx_d = np.asarray(cx_preds[cs:ce])
                    ml = min(len(iy_d), len(cx_d))
                    iy_parts.append(iy_d[:ml])
                    cx_parts.append(cx_d[:ml])

            if iy_parts:
                iy_cat = np.concatenate(iy_parts)
                cx_cat = np.concatenate(cx_parts)
                inter = int(((iy_cat == 1) & (cx_cat == 1)).sum())
                union = int(((iy_cat == 1) | (cx_cat == 1)).sum())
                j = inter / union if union > 0 else 0.0
                n_iy = int((iy_cat == 1).sum())
                n_cx = int((cx_cat == 1).sum())
                n_tot = len(iy_cat)
            else:
                j = inter = union = n_iy = n_cx = n_tot = 0

            jaccard_rows.append(dict(
                target_year=target_year, model=mt, jaccard=j,
                intersection=inter, union=union,
                n_inyear=n_iy, n_crossyear=n_cx, n_samples=n_tot,
            ))
            print(f"  {target_year} test, {MODEL_LABELS[mt]:>8s}: "
                  f"J = {j:.4f}  (inter={inter:,}, union={union:,}, "
                  f"n_iy={n_iy:,}, n_cx={n_cx:,})")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(MODEL_TYPES))
    bw = 0.30

    for i, ty in enumerate(YEARS):
        vals = [d["jaccard"] for d in jaccard_rows if d["target_year"] == ty]
        bars = ax.bar(x + (i - 0.5) * bw, vals, bw,
                      label=f"Target: {ty} test",
                      color=YEAR_COLORS[ty], alpha=0.8,
                      edgecolor="black", linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[mt] for mt in MODEL_TYPES])
    ax.set_ylabel("Jaccard similarity")
    ax.set_title("Detection Overlap: In-Year vs Cross-Year Training")
    ax.legend()

    max_j = max(d["jaccard"] for d in jaccard_rows) if jaccard_rows else 0.1
    ax.set_ylim(0, max(0.1, max_j * 1.3 + 0.02))

    fig.tight_layout()
    save_fig(fig, "fig_5_9_detection_overlap.pdf")

    # ── Summary CSV ─────────────────────────────────────────────────
    rows = []
    for ty in YEARS:
        cy = "2017" if ty == "2015" else "2015"
        for mt in MODEL_TYPES:
            jd = next(d for d in jaccard_rows
                      if d["target_year"] == ty and d["model"] == mt)
            iy_r = rates[ty]["in_year"][mt]
            cx_r = rates[ty]["cross_year"][mt]
            rows.append({
                "train_year": ty,
                "model": MODEL_LABELS[mt],
                "inyear_rate_pct": round(iy_r, 4),
                "crossyear_rate_pct": round(cx_r, 4),
                "cross_target": cy,
                "rate_ratio": round(cx_r / max(iy_r, 1e-6), 3),
                "jaccard_on_test": round(jd["jaccard"], 4),
                "jaccard_inter": jd["intersection"],
                "jaccard_union": jd["union"],
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "cross_year_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  saved {csv_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
