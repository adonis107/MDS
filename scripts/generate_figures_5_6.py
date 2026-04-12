"""
Compute feature attributions and generate figures for Section 5.6.

Applies Integrated Gradients (PNN, PRAE) and Grouped Occlusion (TF-OCSVM)
to a subsample of flagged windows from the test set.  Saves attribution
arrays and generates five figures for the report.

Run from repo root:  python scripts/generate_figures_5_6.py
"""

import os, sys, json, warnings, gc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))

from detection.data.loaders import create_sequences
from detection.trainers.factory import load_model
from detection.sensitivity.integrated_gradients import (
    IntegratedGradients, maximize_sigma, maximize_rec_error,
)
from detection.sensitivity.occlusion import GroupedOcclusion, group_features

# ── Constants ───────────────────────────────────────────────────────
OUT_DIR = os.path.join("figures", "results")
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LENGTH = 25
DEVICE = torch.device("cpu")
N_ATTR_SAMPLES = 150        # flagged windows per model per year

YEARS = ["2015", "2017"]
MODELS_IG = ["pnn", "prae"]

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = (
    "#4878CF", "#E8884A", "#6AB187", "#C44E52", "#999999", "#9B59B6")
MODEL_COLORS  = {"transformer_ocsvm": BLUE, "pnn": ORANGE, "prae": GREEN}
MODEL_LABELS  = {"transformer_ocsvm": "TF\u2013OC-SVM", "pnn": "PNN", "prae": "PRAE"}

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

LOB_COLUMNS = [
    "xltime",
    "bid-price-1", "bid-volume-1", "ask-price-1", "ask-volume-1",
    "bid-price-2", "bid-volume-2", "ask-price-2", "ask-volume-2",
    "bid-price-3", "bid-volume-3", "ask-price-3", "ask-volume-3",
    "bid-price-4", "bid-volume-4", "ask-price-4", "ask-volume-4",
    "bid-price-5", "bid-volume-5", "ask-price-5", "ask-volume-5",
]


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  saved {path}")


# ── Helpers ─────────────────────────────────────────────────────────
def load_feature_names(year):
    with open(f"results/{year}/pnn_features.txt") as f:
        return [line.strip() for line in f if line.strip()]


def get_feature_types(feature_names):
    from detection.sensitivity.occlusion import parse_feature_attributes
    return [parse_feature_attributes(fn)["type"] for fn in feature_names]


def _load_test_meta(year):
    with open(f"results/{year}/test_output/test_meta.json") as f:
        return json.load(f)


def _test_day_indices(meta):
    """Return list of day indices that belong to the test split."""
    return [i for i, l in enumerate(meta["day_split_labels"]) if "test" in l]


def _flagged_in_test(preds, meta):
    """Return global indices of flagged windows that lie in test days."""
    bounds = meta["day_boundaries"]
    test_days = _test_day_indices(meta)
    mask = np.zeros(len(preds), dtype=bool)
    for di in test_days:
        mask[bounds[di]:bounds[di + 1]] = True
    return np.where(mask & (preds == 1))[0]


# ── PNN wrapper for IG (2-D → 3-D adapter) ─────────────────────────
class _PNNIGWrapper(nn.Module):
    """Wraps PNN so IG can pass (batch, 1, feat) tensors."""
    def __init__(self, pnn):
        super().__init__()
        self.pnn = pnn

    def forward(self, x):
        # x: (batch, 1, feat) from IG interpolation
        return self.pnn(x.squeeze(1))


# ── TF-OCSVM assembly for Grouped Occlusion ────────────────────────
class _DetectorShell:
    """Lightweight object with the interface GroupedOcclusion expects."""
    def __init__(self, transformer, ocsvm):
        self.transformer = transformer
        self.ocsvm = ocsvm

    @property
    def device(self):
        return self.transformer.device


# ── Sequence reconstruction ─────────────────────────────────────────
def _load_day_raw_features(path, feature_names):
    """Load a day parquet and return raw feature array (unscaled)."""
    # Read only the columns we need — much faster for 573 MB files
    try:
        df = pd.read_parquet(path, columns=feature_names)
    except Exception:
        # Fallback: some columns may be missing in parquet
        full = pd.read_parquet(path)
        available = [c for c in feature_names if c in full.columns]
        df = full[available].copy()
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df[feature_names]
        del full
    return df.values.astype(np.float32)


def extract_all_models_year(year, feature_names, n_samples, rng):
    """
    For one year: stream through test-day parquet files one at a time,
    greedily collecting flagged windows until n_samples per model are reached.
    Returns per-model {model: (tensor, chosen_indices)} dicts.
    """
    meta = _load_test_meta(year)
    bounds = meta["day_boundaries"]
    day_names = meta["day_names"]
    test_days = _test_day_indices(meta)

    MODEL_TYPES = ["pnn", "prae", "transformer_ocsvm"]

    # 1. Load predictions to know which windows are flagged per model
    model_preds = {}
    for mt in MODEL_TYPES:
        model_preds[mt] = np.load(f"results/{year}/test_output/{mt}_preds.npy")
        flagged_total = _flagged_in_test(model_preds[mt], meta)
        print(f"  {mt}: {len(flagged_total)} flagged test windows")

    # 2. Load scalers
    scalers = {mt: joblib.load(f"results/{year}/{mt}_scaler.pkl") for mt in MODEL_TYPES}

    # 3. Stream through test days, greedily fill from each day's flagged set
    collected = {mt: [] for mt in MODEL_TYPES}
    valid_chosen = {mt: [] for mt in MODEL_TYPES}
    data_dir = "data/processed/TOTF.PA-book"

    for di in test_days:
        all_done = all(len(collected[mt]) >= n_samples for mt in MODEL_TYPES)
        if all_done:
            print("    All models have enough samples, skipping remaining days")
            break

        path = os.path.join(data_dir, day_names[di])
        if not os.path.exists(path):
            print(f"    WARNING: {path} not found")
            continue
        print(f"    {day_names[di]}...", end=" ", flush=True)
        raw = _load_day_raw_features(path, feature_names)
        day_start = bounds[di]
        day_end = bounds[di + 1]
        counts = []

        for mt in MODEL_TYPES:
            still_need = n_samples - len(collected[mt])
            if still_need <= 0:
                continue
            # Find ALL flagged indices in this day
            day_preds = model_preds[mt][day_start:day_end]
            day_flagged = np.where(day_preds == 1)[0]  # local indices within day
            if len(day_flagged) == 0:
                continue
            # Randomly pick up to still_need from this day's flagged set
            pick = rng.choice(day_flagged,
                              size=min(still_need, len(day_flagged)),
                              replace=False)
            pick.sort()
            # Scale and extract windows
            scaled = scalers[mt].transform(raw).astype(np.float32)
            seqs = create_sequences(scaled, SEQ_LENGTH)
            cnt = 0
            for loc in pick:
                if 0 <= loc < len(seqs):
                    collected[mt].append(np.ascontiguousarray(seqs[loc]))
                    valid_chosen[mt].append(day_start + loc)
                    cnt += 1
            counts.append(f"{mt}:{cnt}")
            del scaled, seqs

        del raw; gc.collect()
        print("  ".join(counts))

    # 4. Stack into tensors
    result = {}
    for mt in MODEL_TYPES:
        if collected[mt]:
            result[mt] = (
                torch.tensor(np.stack(collected[mt]), dtype=torch.float32),
                np.array(valid_chosen[mt]),
            )
        else:
            result[mt] = (None, np.array([]))
        print(f"  {mt}: {len(collected[mt])} windows extracted")

    return result


# ── IG computation ──────────────────────────────────────────────────
def compute_pnn_ig(year, seqs, num_features, n_steps=30):
    """IG for PNN.  seqs: (N, 25, num_features). Uses last timestep only."""
    pnn, _ = load_model("pnn", num_features,
                        f"results/{year}/pnn_weights.pth", DEVICE)
    wrapper = _PNNIGWrapper(pnn)
    ig = IntegratedGradients(wrapper)

    attribs = []
    n = len(seqs)
    for i in range(n):
        if i % 50 == 0:
            print(f"    PNN IG: {i}/{n}", end="\r")
        # Last-timestep only, shaped (1, 1, 96) for IG
        x = seqs[i:i + 1, -1:, :]
        try:
            a = ig.attribute(x, target_func=maximize_sigma, n_steps=n_steps)
            attribs.append(a.squeeze().abs().detach().cpu().numpy())
        except Exception:
            attribs.append(np.zeros(num_features))
    print(f"    PNN IG: {n}/{n} done")
    del pnn, wrapper, ig; gc.collect()
    return np.stack(attribs)


def compute_prae_ig(year, seqs, num_features, n_steps=30):
    """IG for PRAE.  seqs: (N, 25, num_features). Full sequence input."""
    prae, _ = load_model("prae", num_features,
                         f"results/{year}/prae_weights.pth", DEVICE,
                         seq_length=SEQ_LENGTH)
    ig = IntegratedGradients(prae)

    attribs = []
    n = len(seqs)
    for i in range(n):
        if i % 50 == 0:
            print(f"    PRAE IG: {i}/{n}", end="\r")
        x = seqs[i:i + 1]
        try:
            a = ig.attribute(x, target_func=maximize_rec_error, n_steps=n_steps)
            # Average |IG| over time → (96,)
            attribs.append(a.squeeze(0).abs().mean(dim=0).detach().cpu().numpy())
        except Exception:
            attribs.append(np.zeros(num_features))
    print(f"    PRAE IG: {n}/{n} done")
    del prae, ig; gc.collect()
    return np.stack(attribs)


def compute_grouped_occlusion(year, seqs, feature_names, num_features):
    """Grouped Occlusion for TF-OCSVM.  Returns DataFrame."""
    transformer, ocsvm = load_model(
        "transformer_ocsvm", num_features,
        f"results/{year}/transformer_ocsvm_weights.pth", DEVICE,
        seq_length=SEQ_LENGTH)
    detector = _DetectorShell(transformer, ocsvm)

    dfs = []
    n = len(seqs)
    for i in range(n):
        if i % 50 == 0:
            print(f"    Occlusion: {i}/{n}", end="\r")
        x = seqs[i:i + 1]
        try:
            imp, _ = GroupedOcclusion(detector, x, feature_names,
                                      group_by="type", baseline_mode="mean")
            imp["sample_idx"] = i
            dfs.append(imp)
        except Exception:
            pass
    print(f"    Occlusion: {n}/{n} done")
    del transformer, ocsvm, detector; gc.collect()
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ── Figure functions ────────────────────────────────────────────────
def fig_ig_bar_charts(ig_results, feature_names, feature_types):
    """Mean |IG| top-20 features for PNN and PRAE, both years."""
    for mt in MODELS_IG:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        for ax, year in zip(axes, YEARS):
            arr = ig_results.get((year, mt))
            if arr is None:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                ax.set_title(f"{MODEL_LABELS[mt]}, {year}")
                continue
            mean = arr.mean(axis=0)
            order = np.argsort(mean)[::-1][:20]
            names = [feature_names[i] for i in order]
            vals  = mean[order]
            colors = [TYPE_COLORS.get(feature_types[i], GREY) for i in order]
            ax.barh(range(len(names)), vals[::-1], color=colors[::-1],
                    edgecolor="white", linewidth=0.3)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names[::-1], fontsize=7)
            ax.set_xlabel("Mean |IG|")
            ax.set_title(f"{MODEL_LABELS[mt]}, {year}")
        fig.tight_layout()
        save_fig(fig, f"fig_5_6_ig_bar_{mt}.pdf")


def fig_ig_variance(ig_results, feature_names, feature_types):
    """Mean vs std of |IG| for PNN flagged windows."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, year in zip(axes, YEARS):
        arr = ig_results.get((year, "pnn"))
        if arr is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            ax.set_title(f"PNN, {year}")
            continue
        means = arr.mean(axis=0)
        stds  = arr.std(axis=0)
        for i in range(len(feature_names)):
            ax.scatter(means[i], stds[i],
                       c=TYPE_COLORS.get(feature_types[i], GREY),
                       s=20, alpha=0.7, edgecolors="white", linewidths=0.3)
        for idx in np.argsort(means)[::-1][:5]:
            ax.annotate(feature_names[idx], (means[idx], stds[idx]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Mean |IG|"); ax.set_ylabel("Std |IG|")
        ax.set_title(f"PNN, {year}")
    fig.tight_layout()
    save_fig(fig, "fig_5_6_ig_variance_pnn.pdf")


def fig_occlusion_bars(occ_results):
    """Grouped Occlusion for TF-OCSVM, both years."""
    fig, axes = plt.subplots(2, 1, figsize=(7, 7))
    for ax, year in zip(axes, YEARS):
        df = occ_results.get(year)
        if df is None or df.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            ax.set_title(f"TF\u2013OC-SVM, {year}")
            continue
        grp = df.groupby("Group")["Importance"].agg(["mean", "std"]).reset_index()
        grp = grp.sort_values("mean", ascending=False).head(15)
        colors = [TYPE_COLORS.get(g, GREY) for g in grp["Group"]]
        ax.barh(range(len(grp)), grp["mean"].values[::-1],
                xerr=grp["std"].values[::-1], capsize=2,
                color=colors[::-1], edgecolor="white", linewidth=0.3)
        ax.set_yticks(range(len(grp)))
        ax.set_yticklabels(grp["Group"].values[::-1], fontsize=8)
        ax.set_xlabel("Mean occlusion importance")
        ax.set_title(f"TF\u2013OC-SVM, {year}")
    fig.tight_layout()
    save_fig(fig, "fig_5_6_occlusion_tf_ocsvm.pdf")


def fig_cross_model_heatmap(ig_results, occ_results, feature_names, feature_types):
    """Normalised cross-model feature-group importance heatmap."""
    type_groups = sorted(set(feature_types))
    type_to_idx = {}
    for i, ft in enumerate(feature_types):
        type_to_idx.setdefault(ft, []).append(i)

    rows = []
    for year in YEARS:
        for m in MODELS_IG:
            arr = ig_results.get((year, m))
            if arr is None: continue
            mean_a = arr.mean(axis=0); total = mean_a.sum()
            if total == 0: continue
            for g in type_groups:
                rows.append({"group": g, "model": m, "year": year,
                             "importance": mean_a[type_to_idx[g]].sum() / total})
        df_occ = occ_results.get(year)
        if df_occ is not None and not df_occ.empty:
            gm = df_occ.groupby("Group")["Importance"].mean()
            total = gm.sum()
            if total > 0:
                for g in type_groups:
                    rows.append({"group": g, "model": "transformer_ocsvm",
                                 "year": year,
                                 "importance": gm.get(g, 0) / total})
    if not rows:
        print("  WARNING: no data for cross-model heatmap"); return
    df = pd.DataFrame(rows)
    df["col"] = df["model"].map(MODEL_LABELS) + " " + df["year"]
    pivot = df.pivot_table(index="group", columns="col",
                           values="importance", fill_value=0)
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=True).drop(columns="total")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if v > 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if v > 0.15 else "black")
    fig.colorbar(im, ax=ax, label="Normalised importance", shrink=0.8)
    ax.set_title("Cross-model feature group importance")
    fig.tight_layout()
    save_fig(fig, "fig_5_6_cross_model_heatmap.pdf")
    pivot.to_csv(os.path.join(OUT_DIR, "cross_model_importance.csv"))
    print(f"  saved cross_model_importance.csv")


def fig_gain_split_ig(ig_results, year, flagged_chosen, feature_names, feature_types):
    """Delta |IG| between high-gain and low-gain flagged windows (PNN)."""
    arr = ig_results.get((year, "pnn"))
    if arr is None: return
    pnn_gain = np.load(f"results/{year}/test_output/pnn_scores.npy")
    chosen = flagged_chosen.get((year, "pnn"))
    if chosen is None or len(chosen) == 0: return
    gains = pnn_gain[chosen[:len(arr)]]
    ok = np.isfinite(gains)
    arr_ok = arr[:len(gains)][ok]
    gains_ok = gains[ok]
    if len(gains_ok) < 10: return

    med = np.median(gains_ok)
    hi = arr_ok[gains_ok >= med].mean(axis=0)
    lo = arr_ok[gains_ok <  med].mean(axis=0)
    diff = hi - lo
    order = np.argsort(np.abs(diff))[::-1][:15]

    fig, ax = plt.subplots(figsize=(7, 5))
    names = [feature_names[i] for i in order]
    vals  = diff[order]
    colors = [TYPE_COLORS.get(feature_types[i], GREY) for i in order]
    ax.barh(range(len(names)), vals[::-1], color=colors[::-1],
            edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=7)
    ax.axvline(0, color="grey", ls="--", lw=0.7)
    ax.set_xlabel(r"$\Delta$ |IG|  (high-gain $-$ low-gain)")
    ax.set_title(f"PNN, {year}: attribution shift by gain level")
    fig.tight_layout()
    save_fig(fig, f"fig_5_6_gain_split_ig_{year}.pdf")


# ── Main ────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(42)
    feature_names = load_feature_names("2015")
    num_features = len(feature_names)
    feature_types = get_feature_types(feature_names)
    print(f"Feature set: {num_features} features")

    ig_results      = {}
    occ_results     = {}
    flagged_chosen  = {}

    for year in YEARS:
        print(f"\n{'='*60}\nYear {year}\n{'='*60}")

        # Extract windows for all 3 models in one pass through day files
        print("\n--- Extracting flagged windows ---")
        extracted = extract_all_models_year(year, feature_names, N_ATTR_SAMPLES, rng)

        # ── PNN IG ──
        print(f"\n--- PNN IG ({year}) ---")
        seqs, chosen = extracted["pnn"]
        flagged_chosen[(year, "pnn")] = chosen
        if seqs is not None:
            ig_results[(year, "pnn")] = compute_pnn_ig(year, seqs, num_features)
        else:
            ig_results[(year, "pnn")] = None

        # ── PRAE IG ──
        print(f"\n--- PRAE IG ({year}) ---")
        seqs, chosen = extracted["prae"]
        flagged_chosen[(year, "prae")] = chosen
        if seqs is not None:
            ig_results[(year, "prae")] = compute_prae_ig(year, seqs, num_features)
        else:
            ig_results[(year, "prae")] = None

        # ── TF-OCSVM Occlusion ──
        print(f"\n--- TF-OCSVM Occlusion ({year}) ---")
        seqs, chosen = extracted["transformer_ocsvm"]
        flagged_chosen[(year, "transformer_ocsvm")] = chosen
        if seqs is not None:
            occ_results[year] = compute_grouped_occlusion(
                year, seqs, feature_names, num_features)
        else:
            occ_results[year] = pd.DataFrame()

        del extracted; gc.collect()

    # ── Save arrays ──
    print("\n--- Saving attribution arrays ---")
    for (yr, mt), arr in ig_results.items():
        if arr is not None:
            p = os.path.join(OUT_DIR, f"ig_attribs_{mt}_{yr}.npy")
            np.save(p, arr); print(f"  saved {p}")
    for yr, df in occ_results.items():
        if not df.empty:
            p = os.path.join(OUT_DIR, f"occlusion_tf_ocsvm_{yr}.csv")
            df.to_csv(p, index=False); print(f"  saved {p}")

    # ── Figures ──
    print("\n--- Generating figures ---")
    fig_ig_bar_charts(ig_results, feature_names, feature_types)
    fig_ig_variance(ig_results, feature_names, feature_types)
    fig_occlusion_bars(occ_results)
    fig_cross_model_heatmap(ig_results, occ_results, feature_names, feature_types)
    for yr in YEARS:
        fig_gain_split_ig(ig_results, yr, flagged_chosen, feature_names, feature_types)

    # ── Summary for LaTeX ──
    print("\n\n=== SUMMARY FOR LATEX ===")
    for (yr, mt), arr in sorted(ig_results.items()):
        if arr is None:
            print(f"\n{yr} {MODEL_LABELS[mt]}: No data"); continue
        mean = arr.mean(axis=0)
        order = np.argsort(mean)[::-1][:10]
        print(f"\n{yr} {MODEL_LABELS[mt]} top-10 features by mean |IG|:")
        for r, idx in enumerate(order, 1):
            print(f"  {r:2d}. {feature_names[idx]:40s} {mean[idx]:.6f}  [{feature_types[idx]}]")
        if mt == "prae":
            print(f"  Sum(mean |IG|): {mean.sum():.4f}")
    for yr in YEARS:
        df = occ_results.get(yr)
        if df is not None and not df.empty:
            g = df.groupby("Group")["Importance"].agg(["mean", "std"]) \
                  .sort_values("mean", ascending=False)
            print(f"\n{yr} TF-OCSVM top groups by mean occlusion importance:")
            for i, (gn, row) in enumerate(g.head(10).iterrows(), 1):
                print(f"  {i:2d}. {gn:20s}  mean={row['mean']:.6f}  std={row['std']:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
