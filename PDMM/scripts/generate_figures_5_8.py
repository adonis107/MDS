"""
Jump analysis for Section 5.8: co-occurrence of flagged windows with
subsequent mid-price jumps, average price path around flags, and lead
time distribution.

A "jump" is a single-event absolute log-return exceeding the 99.95th
percentile of absolute returns across all test events for that year.
Co-occurrence measures whether flagged windows are followed by a jump
within JUMP_HORIZON events more often than the test-set base rate.

Run from repo root:  python scripts/generate_figures_5_8.py
"""

import os, sys, json, warnings, gc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT_DIR = os.path.join("figures", "results")
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LENGTH = 25
YEARS = ["2015", "2017"]
MODEL_TYPES = ["transformer_ocsvm", "pnn", "prae"]
MODEL_LABELS = {
    "transformer_ocsvm": "TF-OCSVM",
    "pnn": "PNN",
    "prae": "PRAE",
}
DATA_DIR = "data/processed/TOTF.PA-book"

JUMP_QUANTILE = 0.9995
JUMP_HORIZON = 100
EVENT_STUDY_HALF = 250
N_PERM = 2000
N_ES_MAX = 5000
N_BOOTSTRAP = 500

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
BLUE, ORANGE, GREEN, RED, GREY = (
    "#4878CF", "#E8884A", "#6AB187", "#C44E52", "#999999")
MODEL_COLORS = {
    "transformer_ocsvm": BLUE, "pnn": ORANGE,
    "prae": GREEN, "consensus": RED,
}


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  saved {path}")


def analyze_year(year):
    print(f"\n{'='*60}\nYear {year}\n{'='*60}")

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
    consensus = n_models == 3
    any_model = n_models >= 1

    print(f"  Test windows: {n_test:,}")
    for mt in MODEL_TYPES:
        print(f"    {MODEL_LABELS[mt]}: {preds[mt].sum():,}")
    print(f"    Consensus (3/3): {consensus.sum():,}")
    print(f"    Any model (>=1): {any_model.sum():,}")

    day_data = []
    all_abs_r = []
    for d in test_days:
        dn = day_names[d]
        df = pd.read_parquet(os.path.join(DATA_DIR, dn),
                             columns=["bid-price-1", "ask-price-1"])
        mid = ((df["bid-price-1"] + df["ask-price-1"]) / 2).values.astype(np.float64)
        r = np.diff(np.log(mid))
        r[~np.isfinite(r)] = 0.0
        n_seqs = bounds[d + 1] - bounds[d]
        local_start = bounds[d] - test_start
        day_data.append({"mid": mid, "r": r, "n_seqs": n_seqs,
                         "local_start": local_start})
        all_abs_r.append(np.abs(r))
        del df

    pooled = np.concatenate(all_abs_r)
    threshold = float(np.percentile(pooled, JUMP_QUANTILE * 100))
    n_jumps_total = int((pooled > threshold).sum())
    n_events_total = len(pooled)
    jump_rate_evt = n_jumps_total / n_events_total
    print(f"  Jump threshold ({JUMP_QUANTILE*100:.2f}th pct): "
          f"{threshold:.4e} ({threshold*1e4:.2f} bps)")
    print(f"  Jump events: {n_jumps_total:,} / {n_events_total:,} "
          f"({100*jump_rate_evt:.4f}%)")
    del all_abs_r, pooled
    gc.collect()

    jump_follows = np.zeros(n_test, dtype=bool)
    first_jump_lag = np.full(n_test, -1, dtype=np.int32)

    for dd in day_data:
        r = dd["r"]
        n_seqs = dd["n_seqs"]
        ls = dd["local_start"]

        J = (np.abs(r) > threshold).astype(np.int32)
        J_pad = np.concatenate([J, np.zeros(JUMP_HORIZON, dtype=np.int32)])
        cs = np.concatenate([[0], np.cumsum(J_pad)])

        anchors = np.arange(SEQ_LENGTH - 1, SEQ_LENGTH - 1 + n_seqs)
        end = np.minimum(anchors + JUMP_HORIZON, len(J))
        has_jump = (cs[end] - cs[anchors]) > 0
        jump_follows[ls:ls + n_seqs] = has_jump

        jpos = np.where(J[:len(r)])[0]
        if len(jpos) > 0:
            wj = np.where(has_jump)[0]
            if len(wj) > 0:
                a_vals = anchors[wj]
                fi = np.searchsorted(jpos, a_vals, side="left")
                fi_c = np.minimum(fi, len(jpos) - 1)
                jp = jpos[fi_c]
                ok = (fi < len(jpos)) & (jp < a_vals + JUMP_HORIZON)
                lags = jp - a_vals
                first_jump_lag[ls + wj[ok]] = lags[ok]

    base_rate = float(jump_follows.mean())
    print(f"  Base rate (jump within {JUMP_HORIZON} events): {base_rate:.4%}")

    rng = np.random.default_rng(42)
    categories = {mt: preds[mt] for mt in MODEL_TYPES}
    categories["consensus"] = consensus

    cooc = {}
    for cat, mask in categories.items():
        n_f = int(mask.sum())
        if n_f == 0:
            cooc[cat] = dict(n_flagged=0, n_with_jump=0, rate=np.nan,
                             p_value=np.nan, null_10=np.nan, null_90=np.nan)
            print(f"    {cat}: n=0, skipped")
            continue
        rate = float(jump_follows[mask].mean())
        n_wj = int(jump_follows[mask].sum())
        null = rng.binomial(n_f, base_rate, size=N_PERM) / n_f
        p_val = max(float((null >= rate).mean()), 1.0 / N_PERM)
        cooc[cat] = dict(
            n_flagged=n_f, n_with_jump=n_wj, rate=rate, p_value=p_val,
            null_10=float(np.percentile(null, 10)),
            null_90=float(np.percentile(null, 90)),
        )
        ratio = rate / base_rate if base_rate > 0 else np.nan
        print(f"    {cat}: n={n_f:,}, rate={rate:.4%}, "
              f"ratio={ratio:.2f}x, p={p_val:.4f}")

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
                if s >= 0 and e <= len(mid):
                    paths.append(np.log(mid[s:e] / mid[a]) * 1e4)
        return np.array(paths) if paths else np.empty((0, PL))

    print(f"  Extracting price paths ({len(es_idx)} flagged, "
          f"{len(nf_sample)} non-flagged)...", flush=True)
    fp = extract_paths(es_idx)
    nfp = extract_paths(nf_sample)
    print(f"    {len(fp)} flagged paths, {len(nfp)} non-flagged paths")

    if len(fp) >= 2:
        mean_f = fp.mean(axis=0)
        bm = np.empty((N_BOOTSTRAP, PL))
        for b in range(N_BOOTSTRAP):
            bm[b] = fp[rng.choice(len(fp), len(fp), replace=True)].mean(axis=0)
        ci_lo = np.percentile(bm, 2.5, axis=0)
        ci_hi = np.percentile(bm, 97.5, axis=0)
    else:
        mean_f = ci_lo = ci_hi = np.full(PL, np.nan)

    mean_nf = nfp.mean(axis=0) if len(nfp) >= 2 else np.full(PL, np.nan)

    lt_mask = any_model & jump_follows
    lead_times = first_jump_lag[lt_mask]
    lead_times = lead_times[lead_times >= 0]
    if len(lead_times) > 0:
        print(f"  Lead time: median={np.median(lead_times):.0f}, "
              f"IQR=[{np.percentile(lead_times, 25):.0f}, "
              f"{np.percentile(lead_times, 75):.0f}]")

    rows = []
    for cat in list(MODEL_TYPES) + ["consensus"]:
        c = cooc[cat]
        rows.append({
            "category": MODEL_LABELS.get(cat, "Consensus"),
            "year": year,
            "n_flagged": c["n_flagged"],
            "n_with_jump": c["n_with_jump"],
            "rate": c["rate"],
            "base_rate": base_rate,
            "ratio": (c["rate"] / base_rate
                      if base_rate > 0 and np.isfinite(c["rate"]) else np.nan),
            "p_value": c["p_value"],
        })
    df_out = pd.DataFrame(rows)
    csv_path = f"results/{year}/test_output/jump_analysis.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"  saved {csv_path}")

    return dict(
        year=year, n_test=n_test, threshold=threshold,
        base_rate=base_rate, jump_rate_evt=jump_rate_evt,
        cooc=cooc, mean_flagged_path=mean_f,
        ci_lo=ci_lo, ci_hi=ci_hi, mean_nonflagged_path=mean_nf,
        n_flagged_paths=len(fp), n_nonflagged_paths=len(nfp),
        lead_times=lead_times,
    )


def fig_cooccurrence(results):
    """Bar chart: fraction of flagged windows followed by a jump."""
    n_years = len(results)
    fig, axes = plt.subplots(n_years, 1, figsize=(7, 4 * n_years),
                             squeeze=False)

    cats = list(MODEL_TYPES) + ["consensus"]
    labels = [MODEL_LABELS.get(c, "Consensus") for c in cats]
    colors = [MODEL_COLORS[c] for c in cats]

    for row, res in enumerate(results):
        ax = axes[row, 0]
        year = res["year"]
        base = res["base_rate"]
        cooc = res["cooc"]

        vals, e_lo, e_hi = [], [], []
        bar_labels, bar_colors = [], []
        for cat, lab, col in zip(cats, labels, colors):
            c = cooc[cat]
            if c["n_flagged"] == 0:
                continue
            vals.append(c["rate"])
            e_lo.append(c["null_10"])
            e_hi.append(c["null_90"])
            bar_labels.append(lab)
            bar_colors.append(col)

        x = np.arange(len(vals))
        ax.bar(x, vals, color=bar_colors, edgecolor="white",
               linewidth=0.5, width=0.6, zorder=3)

        for i in range(len(vals)):
            ax.plot([i, i], [e_lo[i], e_hi[i]],
                    color="black", linewidth=1.5, zorder=4)
            ax.plot([i - 0.08, i + 0.08], [e_lo[i]] * 2,
                    color="black", linewidth=1.5, zorder=4)
            ax.plot([i - 0.08, i + 0.08], [e_hi[i]] * 2,
                    color="black", linewidth=1.5, zorder=4)

        ax.axhline(base, color="black", ls="--", lw=1, zorder=2,
                   label=f"Base rate = {base:.3%}")

        active_cats = [c for c in cats if cooc[c]["n_flagged"] > 0]
        for i, cat in enumerate(active_cats):
            c = cooc[cat]
            ratio = c["rate"] / base if base > 0 else np.nan
            p = c["p_value"]
            p_str = (f"p < {1/N_PERM:.0e}" if p <= 1.0 / N_PERM
                     else f"p = {p:.3f}")
            ax.text(i, vals[i] + 0.001, f"{vals[i]:.3%}\n({ratio:.1f}x, {p_str})",
                    ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels)
        ax.set_ylabel("Co-occurrence rate")
        ax.set_title(f"{year}: flagged windows followed by a jump "
                     f"within {JUMP_HORIZON} events")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "fig_5_8_cooccurrence.pdf")


def fig_price_path(results):
    """Average mid-price trajectory around flagged windows."""
    n_years = len(results)
    fig, axes = plt.subplots(n_years, 1, figsize=(8, 4 * n_years),
                             squeeze=False)
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
            ax.fill_between(t, ci_lo, ci_hi,
                            color=BLUE, alpha=0.2, zorder=2)

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


def fig_lead_time(results):
    """Histogram of lag between flagged window and next jump."""
    n_years = len(results)
    fig, axes = plt.subplots(n_years, 1, figsize=(7, 4 * n_years),
                             squeeze=False)

    for row, res in enumerate(results):
        ax = axes[row, 0]
        year = res["year"]
        lt = res["lead_times"]

        if len(lt) == 0:
            ax.text(0.5, 0.5, "No jumps following flagged windows",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{year}: lead time distribution")
            continue

        bins = np.arange(0, JUMP_HORIZON + 2) - 0.5
        ax.hist(lt, bins=bins, color=BLUE, edgecolor="white",
                linewidth=0.3, alpha=0.8)

        med = np.median(lt)
        q25, q75 = np.percentile(lt, [25, 75])
        ax.axvline(med, color=RED, ls="--", lw=1.2,
                   label=f"Median = {med:.0f}")
        ax.axvspan(q25, q75, color=ORANGE, alpha=0.15,
                   label=f"IQR = [{q25:.0f}, {q75:.0f}]")

        ax.set_xlabel("Lag (events after anchor)")
        ax.set_ylabel("Count")
        ax.set_title(f"{year}: lag from flagged window to next jump "
                     f"(n = {len(lt):,})")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "fig_5_8_lead_time.pdf")


def main():
    results = []
    for year in YEARS:
        res = analyze_year(year)
        results.append(res)
        gc.collect()

    print("\n--- Generating figures ---")
    fig_cooccurrence(results)
    fig_price_path(results)
    fig_lead_time(results)

    print("\n=== SUMMARY FOR LATEX ===")
    for res in results:
        yr = res["year"]
        print(f"\n{yr}:")
        print(f"  Jump threshold: {res['threshold']:.4e} "
              f"({res['threshold']*1e4:.2f} bps)")
        print(f"  Base rate: {res['base_rate']:.4%}")
        cooc = res["cooc"]
        for cat in list(MODEL_TYPES) + ["consensus"]:
            c = cooc[cat]
            if c["n_flagged"] == 0:
                print(f"  {MODEL_LABELS.get(cat, 'Consensus')}: n=0")
                continue
            ratio = c["rate"] / res["base_rate"] if res["base_rate"] > 0 else np.nan
            print(f"  {MODEL_LABELS.get(cat, 'Consensus')}: "
                  f"n={c['n_flagged']:,}, rate={c['rate']:.4%}, "
                  f"ratio={ratio:.2f}x, p={c['p_value']:.4f}")
        lt = res["lead_times"]
        if len(lt) > 0:
            print(f"  Lead time: n={len(lt):,}, median={np.median(lt):.0f}, "
                  f"IQR=[{np.percentile(lt, 25):.0f}, "
                  f"{np.percentile(lt, 75):.0f}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
