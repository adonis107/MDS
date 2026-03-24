import argparse
import errno
import glob
import json
import logging
import os
import re
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from detection.data.loaders import create_sequences, load_processed
from detection.data.preprocessing import assign_period, get_time_frac
from detection.models import prae
from detection.spoofing.gain import compute_spoofing_gains_batch
from detection.trainers.factory import load_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_inference_cache")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TRAIN_YEAR = "2015"
MODEL_TYPES = ["transformer_ocsvm", "pnn", "prae"]
DEFAULT_PATTERNS = [
    "2009-*",
    "2010-*",
    "2015-02-03*",
    "2015-02-04*",
    "2015-02-05*",
    "2017-*",
]

LOB_COLUMNS = [
    f"{side}-{typ}-{lvl}"
    for lvl in range(1, 11)
    for side, typ in [
        ("bid", "price"),
        ("bid", "volume"),
        ("ask", "price"),
        ("ask", "volume"),
    ]
]

PERIODS = {
    "1st_hour": (9.0, 10.0),
    "rest_of_morning": (10.0, 12.0),
    "afternoon": (12.0, 15.5),
    "american_open": (15.5, 17.5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run test-time inference once and cache all model-stage outputs per day."
    )
    parser.add_argument("--train-year", default=TRAIN_YEAR, help="Model/scaler year in results/{year}.")
    parser.add_argument(
        "--data-dir",
        default=os.path.join("data", "processed", "TOTF.PA-book"),
        help="Directory with processed parquet files.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root results directory.",
    )
    parser.add_argument(
        "--cache-subdir",
        default="test_cache",
        help="Cache sub-directory inside results/{train_year}.",
    )
    parser.add_argument(
        "--test-file-patterns",
        nargs="+",
        default=DEFAULT_PATTERNS,
        help="Glob patterns matched against --data-dir.",
    )
    parser.add_argument("--seq-length", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true", help="Overwrite existing day cache.")

    # Gain parameters for PNN-derived score stream
    parser.add_argument("--spoof-q-big", type=float, default=4500.0)
    parser.add_argument("--spoof-q-small", type=float, default=100.0)
    parser.add_argument("--spoof-delta-a", type=float, default=0.0)
    parser.add_argument("--spoof-delta-b", type=float, default=0.01)
    parser.add_argument("--spoof-maker-fee", type=float, default=0.0)
    parser.add_argument("--spoof-taker-fee", type=float, default=0.05)

    return parser.parse_args()


def _safe_day_key(index: int, filepath: str) -> str:
    base = os.path.splitext(os.path.basename(filepath))[0]
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", base)
    return f"{index:03d}_{base}"


def resolve_test_files(data_dir: str, patterns: List[str]) -> List[str]:
    matched = set()
    for pattern in patterns:
        with_ext = sorted(glob.glob(os.path.join(data_dir, pattern + ".parquet")))
        if with_ext:
            matched.update(with_ext)
            continue
        raw = sorted(glob.glob(os.path.join(data_dir, pattern)))
        matched.update(raw)

    test_files = sorted(matched)
    if not test_files:
        raise FileNotFoundError(f"No test files found for patterns: {patterns}")
    return test_files


def load_feature_names(results_dir_for_year: str, test_files: List[str]) -> Dict[str, List[str]]:
    feature_names_map: Dict[str, List[str]] = {}

    for model_type in MODEL_TYPES:
        feat_path = os.path.join(results_dir_for_year, f"{model_type}_features.txt")
        if os.path.exists(feat_path):
            with open(feat_path, "r", encoding="utf-8") as f:
                feature_names_map[model_type] = [line.strip() for line in f if line.strip()]
        else:
            _, feat_tmp = load_processed(test_files[0], "xltime", LOB_COLUMNS)
            feature_names_map[model_type] = feat_tmp.columns.tolist()

    return feature_names_map


def load_models_and_scalers(
    results_dir_for_year: str,
    seq_length: int,
    feature_names_map: Dict[str, List[str]],
) -> Tuple[Dict[str, Tuple[torch.nn.Module, object]], Dict[str, object]]:
    loaded_models: Dict[str, Tuple[torch.nn.Module, object]] = {}
    loaded_scalers: Dict[str, object] = {}

    for model_type in MODEL_TYPES:
        feat_names = feature_names_map[model_type]
        num_features = len(feat_names)

        weights_path = os.path.join(results_dir_for_year, f"{model_type}_weights.pth")
        model, detector_or_none = load_model(
            model_type,
            num_features,
            weights_path,
            DEVICE,
            seq_length,
        )
        loaded_models[model_type] = (model, detector_or_none)

        scaler_path = os.path.join(results_dir_for_year, f"{model_type}_scaler.pkl")
        loaded_scalers[model_type] = (
            joblib.load(scaler_path) if os.path.exists(scaler_path) else MinMaxScaler()
        )

        logger.info("Loaded %s model (%d features)", model_type, num_features)

    return loaded_models, loaded_scalers


def compute_transformer_stage(
    model: torch.nn.Module,
    detector_or_none: object,
    sequences: np.ndarray,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    x_tensor = torch.tensor(sequences, dtype=torch.float32)

    latents = []
    recon_err = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_tensor), batch_size):
            end = min(start + batch_size, len(x_tensor))
            x = x_tensor[start:end].to(DEVICE)

            z = model.get_representation(x)
            latents.append(z.cpu().numpy())

            rec = model(x)
            err = torch.mean((x - rec) ** 2, dim=(1, 2))
            recon_err.append(err.cpu().numpy())

    latent_arr = np.concatenate(latents, axis=0) if latents else np.empty((0, 0), dtype=np.float32)
    recon_arr = np.concatenate(recon_err, axis=0) if recon_err else np.empty((0,), dtype=np.float32)

    if detector_or_none is not None and len(latent_arr) > 0:
        ocsvm_score = -detector_or_none.decision_function(latent_arr)
    else:
        ocsvm_score = recon_arr.copy()

    return {
        "latent": latent_arr.astype(np.float32),
        "reconstruction_error": recon_arr.astype(np.float32),
        "ocsvm_score": np.asarray(ocsvm_score, dtype=np.float32),
    }


def compute_pnn_stage(
    model: torch.nn.Module,
    sequences: np.ndarray,
    spread_seq: np.ndarray,
    batch_size: int,
    gain_cfg: Dict[str, float],
) -> Dict[str, np.ndarray]:
    all_mu, all_sigma, all_alpha = [], [], []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            end = min(start + batch_size, len(sequences))
            x_batch = torch.tensor(
                np.ascontiguousarray(sequences[start:end]),
                dtype=torch.float32,
            ).reshape(end - start, -1).to(DEVICE)
            mu, sigma, alpha = model(x_batch)
            all_mu.append(mu.cpu().numpy().flatten())
            all_sigma.append(sigma.cpu().numpy().flatten())
            all_alpha.append(alpha.cpu().numpy().flatten())

    mu_arr = np.concatenate(all_mu) if all_mu else np.empty((0,), dtype=np.float32)
    sigma_arr = np.concatenate(all_sigma) if all_sigma else np.empty((0,), dtype=np.float32)
    alpha_arr = np.concatenate(all_alpha) if all_alpha else np.empty((0,), dtype=np.float32)

    spread_seq = spread_seq[: len(mu_arr)]
    if len(spread_seq) < len(mu_arr):
        spread_seq = np.pad(spread_seq, (0, len(mu_arr) - len(spread_seq)), mode="edge")

    spread_seq = np.abs(spread_seq)
    spread_seq = np.where(spread_seq > 0, spread_seq, 1e-4)

    gain_score = compute_spoofing_gains_batch(
        mu_arr,
        sigma_arr,
        alpha_arr,
        spread_seq,
        delta_a=gain_cfg["delta_a"],
        delta_b=gain_cfg["delta_b"],
        Q=gain_cfg["Q"],
        q=gain_cfg["q"],
        fees=gain_cfg["fees"],
        side="ask",
    )

    return {
        "mu": np.asarray(mu_arr, dtype=np.float32),
        "sigma": np.asarray(sigma_arr, dtype=np.float32),
        "alpha": np.asarray(alpha_arr, dtype=np.float32),
        "spread": np.asarray(spread_seq, dtype=np.float32),
        "gain_score": np.asarray(gain_score, dtype=np.float32),
    }


def compute_prae_stage(
    model: torch.nn.Module,
    sequences: np.ndarray,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    x_tensor = torch.tensor(sequences, dtype=torch.float32)

    score_parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_tensor), batch_size):
            end = min(start + batch_size, len(x_tensor))
            x = x_tensor[start:end].to(DEVICE)
            rec, _ = model(x, training=False)
            err = torch.sum((x - rec) ** 2, dim=tuple(range(1, x.dim())))
            score_parts.append(err.cpu().numpy())

    reconstruction_error = (
        np.concatenate(score_parts) if score_parts else np.empty((0,), dtype=np.float32)
    )

    return {
        "reconstruction_error": np.asarray(reconstruction_error, dtype=np.float32),
    }


def savez_compressed_atomic(path: str, **arrays: np.ndarray) -> None:
    """Write a compressed npz atomically to avoid leaving corrupted partial files."""
    tmp_path = f"{path}.tmp"
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, path)
    except OSError as exc:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        if exc.errno in (errno.EDQUOT, errno.ENOSPC):
            raise RuntimeError(
                "Disk quota / space exceeded while writing cache file "
                f"{path!r}. Free space or write to a scratch filesystem "
                "(--results-dir), then relaunch — already-completed days "
                "will be skipped automatically."
            ) from exc
        raise


def main() -> None:
    args = parse_args()
    logger.info("Device: %s", DEVICE)

    results_dir_for_year = os.path.join(args.results_dir, str(args.train_year))
    cache_root = os.path.join(results_dir_for_year, args.cache_subdir)
    days_root = os.path.join(cache_root, "days")
    os.makedirs(days_root, exist_ok=True)

    test_files = resolve_test_files(args.data_dir, args.test_file_patterns)
    logger.info("Found %d test files", len(test_files))

    feature_names_map = load_feature_names(results_dir_for_year, test_files)
    loaded_models, loaded_scalers = load_models_and_scalers(
        results_dir_for_year,
        args.seq_length,
        feature_names_map,
    )

    gain_cfg = {
        "Q": args.spoof_q_big,
        "q": args.spoof_q_small,
        "delta_a": args.spoof_delta_a,
        "delta_b": args.spoof_delta_b,
        "fees": {
            "maker": args.spoof_maker_fee,
            "taker": args.spoof_taker_fee,
        },
    }

    run_manifest = {
        "train_year": str(args.train_year),
        "data_dir": args.data_dir,
        "seq_length": int(args.seq_length),
        "batch_size": int(args.batch_size),
        "periods": PERIODS,
        "test_file_patterns": args.test_file_patterns,
        "test_files": [os.path.basename(f) for f in test_files],
        "gain_config": gain_cfg,
        "models": MODEL_TYPES,
        "days": [],
    }

    day_boundaries = [0]
    total_sequences = 0

    for file_idx, test_file in enumerate(test_files):
        day_key = _safe_day_key(file_idx, test_file)
        day_dir = os.path.join(days_root, day_key)
        os.makedirs(day_dir, exist_ok=True)

        complete_flag = os.path.join(day_dir, "_COMPLETE.json")
        if os.path.exists(complete_flag) and not args.force:
            with open(complete_flag, "r", encoding="utf-8") as f:
                complete_meta = json.load(f)
            n_seq_day = int(complete_meta.get("n_seq", 0))
            total_sequences += n_seq_day
            day_boundaries.append(total_sequences)
            run_manifest["days"].append(complete_meta)
            logger.info("Skipping cached day %s (%d seq)", day_key, n_seq_day)
            continue

        logger.info("Processing %s", os.path.basename(test_file))

        df_day, features_day = load_processed(test_file, "xltime", LOB_COLUMNS)
        n_seq_day = len(features_day) - args.seq_length
        if n_seq_day <= 0:
            logger.warning("Skipping %s (not enough rows for sequences)", test_file)
            continue

        time_frac_day = get_time_frac(df_day)[: len(features_day)]
        period_labels_day = assign_period(time_frac_day, PERIODS)
        period_labels_seq = period_labels_day[args.seq_length : args.seq_length + n_seq_day]

        feature_values_seq = (
            features_day.iloc[args.seq_length : args.seq_length + n_seq_day]
            .reset_index(drop=True)
            .to_numpy(dtype=np.float32)
        )
        np.save(os.path.join(day_dir, "feature_values_seq.npy"), feature_values_seq)

        with open(os.path.join(day_dir, "feature_columns.json"), "w", encoding="utf-8") as f:
            json.dump({"columns": features_day.columns.tolist()}, f, indent=2)

        savez_compressed_atomic(
            os.path.join(day_dir, "common.npz"),
            period_labels=np.asarray(period_labels_seq, dtype="<U32"),
        )

        spread_raw_day = (df_day["ask-price-1"] - df_day["bid-price-1"]).to_numpy()

        day_model_summary = {}

        for model_type in MODEL_TYPES:
            feat_names = feature_names_map[model_type]
            scaler = loaded_scalers[model_type]
            model, detector_or_none = loaded_models[model_type]

            feat_df = features_day.copy()
            for col in feat_names:
                if col not in feat_df.columns:
                    feat_df[col] = 0.0
            feat_df = feat_df[feat_names]

            scaled = scaler.transform(feat_df.to_numpy(dtype=np.float32)).astype(np.float32)
            sequences = create_sequences(scaled, args.seq_length)

            if model_type == "transformer_ocsvm":
                stage = compute_transformer_stage(
                    model=model,
                    detector_or_none=detector_or_none,
                    sequences=sequences,
                    batch_size=args.batch_size,
                )
                savez_compressed_atomic(
                    os.path.join(day_dir, "transformer_ocsvm_stage.npz"),
                    latent=stage["latent"],
                    reconstruction_error=stage["reconstruction_error"],
                    ocsvm_score=stage["ocsvm_score"],
                )

            elif model_type == "pnn":
                spread_seq = spread_raw_day[args.seq_length : args.seq_length + len(sequences)]
                stage = compute_pnn_stage(
                    model=model,
                    sequences=sequences,
                    spread_seq=spread_seq,
                    batch_size=args.batch_size,
                    gain_cfg=gain_cfg,
                )
                savez_compressed_atomic(
                    os.path.join(day_dir, "pnn_stage.npz"),
                    mu=stage["mu"],
                    sigma=stage["sigma"],
                    alpha=stage["alpha"],
                    spread=stage["spread"],
                    gain_score=stage["gain_score"],
                )

            elif model_type == "prae":
                stage = compute_prae_stage(
                    model=model,
                    sequences=sequences,
                    batch_size=args.batch_size,
                )
                savez_compressed_atomic(
                    os.path.join(day_dir, "prae_stage.npz"),
                    reconstruction_error=stage["reconstruction_error"],
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            day_model_summary[model_type] = {
                "n_sequences": int(len(sequences)),
                "n_features": int(len(feat_names)),
            }

        day_meta = {
            "day_key": day_key,
            "source_file": os.path.basename(test_file),
            "n_rows": int(len(features_day)),
            "n_seq": int(n_seq_day),
            "model_summary": day_model_summary,
        }

        with open(complete_flag, "w", encoding="utf-8") as f:
            json.dump(day_meta, f, indent=2)

        run_manifest["days"].append(day_meta)

        total_sequences += n_seq_day
        day_boundaries.append(total_sequences)

        logger.info("Saved cache for %s (%d sequences)", day_key, n_seq_day)

    run_manifest["total_sequences"] = int(total_sequences)
    run_manifest["day_boundaries"] = day_boundaries

    manifest_path = os.path.join(cache_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    logger.info("=" * 70)
    logger.info("Inference cache complete")
    logger.info("Cache root: %s", cache_root)
    logger.info("Manifest: %s", manifest_path)
    logger.info("Total sequences: %d", total_sequences)


if __name__ == "__main__":
    main()
