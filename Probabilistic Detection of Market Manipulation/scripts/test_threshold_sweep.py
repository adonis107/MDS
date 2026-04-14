import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from detection.thresholds.dspot import DriftStreamingPeakOverThreshold
from detection.thresholds.pot import PeakOverThreshold
from detection.thresholds.rfdr import RollingFalseDiscoveryRate


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_threshold_sweep")


TRAIN_YEAR = "2015"
DEFAULT_METHODS = ["pot", "dspot", "rfdr"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run threshold sweeps from cached test inference artifacts (no model rerun)."
    )
    parser.add_argument("--train-year", default=TRAIN_YEAR)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--cache-subdir", default="test_cache")
    parser.add_argument("--threshold-subdir", default="threshold_runs")
    parser.add_argument(
        "--run-tag",
        default=datetime.now().strftime("run_%Y%m%d_%H%M%S"),
        help="Output run id under threshold_subdir.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=["pot", "dspot", "rfdr"],
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        default=["all"],
        help="Score stream names, or 'all'.",
    )

    parser.add_argument("--pot-risk", type=float, default=1e-4)
    parser.add_argument("--pot-init-level", type=float, default=0.98)
    parser.add_argument("--pot-num-candidates", type=int, default=10)
    parser.add_argument("--pot-epsilon", type=float, default=1e-8)

    parser.add_argument("--dspot-risk", type=float, default=1e-4)
    parser.add_argument("--dspot-init-level", type=float, default=0.98)
    parser.add_argument("--dspot-num-candidates", type=int, default=10)
    parser.add_argument("--dspot-epsilon", type=float, default=1e-8)
    parser.add_argument("--dspot-num-init", type=int, default=0, help="0 means auto")
    parser.add_argument("--dspot-depth", type=int, default=50)

    parser.add_argument("--rfdr-window", type=int, default=500)
    parser.add_argument("--rfdr-alpha", type=float, default=0.05)

    return parser.parse_args()


def _load_manifest(cache_root: str) -> Dict:
    manifest_path = os.path.join(cache_root, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Cache manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_streams_for_day(day_dir: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    common = np.load(os.path.join(day_dir, "common.npz"), allow_pickle=False)
    period_labels = common["period_labels"]

    t_stage = np.load(os.path.join(day_dir, "transformer_ocsvm_stage.npz"), allow_pickle=False)
    p_stage = np.load(os.path.join(day_dir, "pnn_stage.npz"), allow_pickle=False)
    r_stage = np.load(os.path.join(day_dir, "prae_stage.npz"), allow_pickle=False)

    streams = {
        "transformer_reconstruction_error": t_stage["reconstruction_error"].astype(np.float32),
        "transformer_ocsvm_score": t_stage["ocsvm_score"].astype(np.float32),
        "pnn_gain_score": p_stage["gain_score"].astype(np.float32),
        "pnn_mu": p_stage["mu"].astype(np.float32),
        "pnn_sigma": p_stage["sigma"].astype(np.float32),
        "pnn_alpha": p_stage["alpha"].astype(np.float32),
        "prae_reconstruction_error": r_stage["reconstruction_error"].astype(np.float32),
    }

    return streams, period_labels


def _concat_streams(cache_root: str, manifest: Dict) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[int], List[str]]:
    days_root = os.path.join(cache_root, "days")

    stream_parts: Dict[str, List[np.ndarray]] = {}
    period_parts: List[np.ndarray] = []
    day_boundaries = [0]
    day_names: List[str] = []

    for day in manifest["days"]:
        day_key = day["day_key"]
        day_dir = os.path.join(days_root, day_key)
        streams, period_labels = _load_streams_for_day(day_dir)

        for name, arr in streams.items():
            stream_parts.setdefault(name, []).append(arr)

        period_parts.append(period_labels)
        day_names.append(day.get("source_file", day_key))

        n_seq = int(day.get("n_seq", len(period_labels)))
        day_boundaries.append(day_boundaries[-1] + n_seq)

    concat_streams = {
        name: np.concatenate(parts) if parts else np.empty((0,), dtype=np.float32)
        for name, parts in stream_parts.items()
    }
    period_labels = np.concatenate(period_parts) if period_parts else np.empty((0,), dtype="<U32")

    if concat_streams:
        min_len = min(len(arr) for arr in concat_streams.values())
        concat_streams = {k: v[:min_len] for k, v in concat_streams.items()}
        period_labels = period_labels[:min_len]

        clipped = [0]
        for b in day_boundaries[1:]:
            clipped.append(min(b, min_len))
            if clipped[-1] == min_len:
                break
        day_boundaries = clipped

    return concat_streams, period_labels, day_boundaries, day_names


def _apply_pot(scores: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Dict]:
    z, t = PeakOverThreshold(
        data=scores,
        num_candidates=args.pot_num_candidates,
        risk=args.pot_risk,
        init_level=args.pot_init_level,
        epsilon=args.pot_epsilon,
    )
    thresholds = np.full(len(scores), float(z), dtype=np.float32)
    preds = (scores > thresholds).astype(np.int32)
    meta = {
        "threshold": float(z),
        "initial_threshold": float(t),
        "risk": float(args.pot_risk),
        "init_level": float(args.pot_init_level),
        "num_candidates": int(args.pot_num_candidates),
        "epsilon": float(args.pot_epsilon),
    }
    return preds, thresholds, meta


def _resolve_dspot_num_init(n: int, requested: int, depth: int) -> int:
    if requested > 0:
        num_init = requested
    else:
        num_init = max(20, n // 2)

    max_allowed = max(1, n - depth - 1)
    return max(1, min(num_init, max_allowed))


def _apply_dspot(scores: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Dict]:
    num_init = _resolve_dspot_num_init(len(scores), args.dspot_num_init, args.dspot_depth)
    thresholds = DriftStreamingPeakOverThreshold(
        data=scores,
        num_init=num_init,
        depth=args.dspot_depth,
        num_candidates=args.dspot_num_candidates,
        risk=args.dspot_risk,
        init_level=args.dspot_init_level,
        epsilon=args.dspot_epsilon,
    ).astype(np.float32)

    thresholds = thresholds[: len(scores)]
    preds = (scores > thresholds).astype(np.int32)

    meta = {
        "num_init": int(num_init),
        "depth": int(args.dspot_depth),
        "risk": float(args.dspot_risk),
        "init_level": float(args.dspot_init_level),
        "num_candidates": int(args.dspot_num_candidates),
        "epsilon": float(args.dspot_epsilon),
    }
    return preds, thresholds, meta


def _apply_rfdr(scores: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Dict]:
    rfdr = RollingFalseDiscoveryRate(window_size=args.rfdr_window, alpha=args.rfdr_alpha)

    preds = np.zeros(len(scores), dtype=np.int32)
    thresholds = np.zeros(len(scores), dtype=np.float32)

    for i, s in enumerate(scores):
        is_anom, threshold = rfdr.process_new_score(float(s))
        preds[i] = int(is_anom)
        thresholds[i] = float(threshold)

    meta = {
        "window_size": int(args.rfdr_window),
        "alpha": float(args.rfdr_alpha),
    }
    return preds, thresholds, meta


def _summarize_by_period(preds: np.ndarray, periods: np.ndarray) -> List[Dict]:
    rows = []
    n = min(len(preds), len(periods))
    if n == 0:
        return rows

    for period_name in sorted(set(periods[:n].tolist())):
        mask = periods[:n] == period_name
        total = int(mask.sum())
        if total == 0:
            continue
        n_anom = int(preds[:n][mask].sum())
        rows.append(
            {
                "period": str(period_name),
                "total": total,
                "anomalies": n_anom,
                "rate_pct": round(100.0 * n_anom / total, 6),
            }
        )
    return rows


def _summarize_by_day(preds: np.ndarray, day_boundaries: List[int], day_names: List[str]) -> List[Dict]:
    rows = []
    n_days = min(len(day_names), max(0, len(day_boundaries) - 1))
    for i in range(n_days):
        lo = min(day_boundaries[i], len(preds))
        hi = min(day_boundaries[i + 1], len(preds))
        n_day = max(0, hi - lo)
        if n_day == 0:
            continue
        day_preds = preds[lo:hi]
        n_anom = int(day_preds.sum())
        rows.append(
            {
                "day": day_names[i],
                "n_samples": n_day,
                "n_anomalies": n_anom,
                "rate_pct": round(100.0 * n_anom / n_day, 6),
            }
        )
    return rows


def _write_csv(rows: List[Dict], out_path: str) -> None:
    if not rows:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _warning_for_stream(name: str) -> str:
    if name in {"pnn_mu", "pnn_sigma", "pnn_alpha"}:
        return "Thresholding raw PNN parameters is exploratory and may not map directly to anomaly semantics."
    if name == "transformer_reconstruction_error":
        return "Transformer reconstruction error thresholding is exploratory when OC-SVM is the primary detector."
    return ""


def main() -> None:
    args = parse_args()

    results_dir_for_year = os.path.join(args.results_dir, str(args.train_year))
    cache_root = os.path.join(results_dir_for_year, args.cache_subdir)
    threshold_root = os.path.join(results_dir_for_year, args.threshold_subdir, args.run_tag)
    os.makedirs(threshold_root, exist_ok=True)

    manifest = _load_manifest(cache_root)
    streams, period_labels, day_boundaries, day_names = _concat_streams(cache_root, manifest)

    if not streams:
        raise RuntimeError("No score streams found in cache.")

    selected_streams = sorted(streams.keys()) if args.streams == ["all"] else args.streams
    unknown = [s for s in selected_streams if s not in streams]
    if unknown:
        raise ValueError(f"Unknown streams requested: {unknown}")

    run_summary = {
        "train_year": str(args.train_year),
        "cache_root": cache_root,
        "run_tag": args.run_tag,
        "methods": args.methods,
        "streams": selected_streams,
        "total_samples": int(min(len(v) for v in streams.values())),
        "outputs": [],
    }

    for stream_name in selected_streams:
        scores = streams[stream_name]
        if len(scores) < 25:
            logger.warning("Skipping %s (too few samples: %d)", stream_name, len(scores))
            continue

        stream_dir = os.path.join(threshold_root, stream_name)
        os.makedirs(stream_dir, exist_ok=True)
        np.save(os.path.join(stream_dir, "scores.npy"), scores.astype(np.float32))
        np.save(os.path.join(stream_dir, "period_labels.npy"), period_labels)
        np.save(os.path.join(stream_dir, "day_boundaries.npy"), np.asarray(day_boundaries, dtype=np.int32))

        for method in args.methods:
            logger.info("Running %s on %s", method, stream_name)

            if method == "pot":
                preds, thresholds, method_meta = _apply_pot(scores, args)
            elif method == "dspot":
                preds, thresholds, method_meta = _apply_dspot(scores, args)
            elif method == "rfdr":
                preds, thresholds, method_meta = _apply_rfdr(scores, args)
            else:
                raise ValueError(f"Unsupported method: {method}")

            out_dir = os.path.join(stream_dir, method)
            os.makedirs(out_dir, exist_ok=True)

            np.save(os.path.join(out_dir, "preds.npy"), preds.astype(np.int32))
            np.save(os.path.join(out_dir, "thresholds.npy"), thresholds.astype(np.float32))

            by_period = _summarize_by_period(preds, period_labels)
            by_day = _summarize_by_day(preds, day_boundaries, day_names)
            _write_csv(by_period, os.path.join(out_dir, "anomaly_rates_by_period.csv"))
            _write_csv(by_day, os.path.join(out_dir, "anomaly_rates_by_day.csv"))

            meta = {
                "stream": stream_name,
                "method": method,
                "n_samples": int(len(scores)),
                "n_anomalies": int(preds.sum()),
                "anomaly_rate_pct": round(100.0 * float(preds.mean()), 6),
                "method_params": method_meta,
                "interpretation_warning": _warning_for_stream(stream_name),
            }
            with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            run_summary["outputs"].append(
                {
                    "stream": stream_name,
                    "method": method,
                    "output_dir": out_dir,
                    "n_samples": int(len(scores)),
                    "n_anomalies": int(preds.sum()),
                }
            )

    with open(os.path.join(threshold_root, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    logger.info("=" * 70)
    logger.info("Threshold sweep complete")
    logger.info("Run output: %s", threshold_root)
    logger.info("Total outputs: %d", len(run_summary["outputs"]))


if __name__ == "__main__":
    main()
