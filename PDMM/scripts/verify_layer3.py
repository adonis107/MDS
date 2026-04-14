"""
Layer 3: Integration check.  
Loads trained models and runs inference on one real data day.
"""
import sys, os, warnings, traceback, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import joblib

RESULTS = []

def record(check_id, status, description, result, action="none required"):
    RESULTS.append((check_id, status, description, result, action))
    tag = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARNING": "[WARN]"}[status]
    print(f"{tag} Check {check_id} -- {description}")
    if status != "PASS":
        print(f"       Result: {result}")
        if action != "none required":
            print(f"       Action: {action}")


def check_3_1():
    """Verify that preprocessed data exists and is well-formed."""
    DATA_DIR = os.path.join("data", "processed", "TOTF.PA-book")
    test_file = None
    for f in sorted(os.listdir(DATA_DIR)):
        if f.startswith("2015") and f.endswith(".parquet"):
            test_file = os.path.join(DATA_DIR, f)
            break
    if test_file is None:
        record("3.1a", "FAIL", "Find processed test file", "No 2015 parquet found")
        return None

    df = pd.read_parquet(test_file)
    if df.shape[0] == 0:
        record("3.1a", "FAIL", "Processed file size",
               f"{test_file}: empty file")
        return None

    record("3.1a", "PASS", "Processed file loaded",
           f"{os.path.basename(test_file)}: {df.shape[0]} rows, {df.shape[1]} cols")

    if df.shape[0] < 1000:
        record("3.1b", "WARNING", "Processed file row count",
               f"Only {df.shape[0]} rows (expected >1000 for a trading day)")
    else:
        record("3.1b", "PASS", "Processed file row count",
               f"{df.shape[0]} rows")

    return test_file


def check_3_2_3(test_file):
    """Load trained 2015 models and run inference on processed data."""
    from detection.data.loaders import create_sequences, load_processed
    from detection.trainers.factory import load_model
    from detection.models.ocsvm import OCSVM

    RESULTS_DIR = os.path.join("results", "2015")
    DEVICE = torch.device("cpu")
    SEQ_LENGTH = 25
    TIME_COL = "xltime"
    LOB_COLUMNS = [
        f"{side}-{typ}-{lvl}"
        for lvl in range(1, 11)
        for side, typ in [("bid","price"),("bid","volume"),("ask","price"),("ask","volume")]
    ]

    feat_path = os.path.join(RESULTS_DIR, "transformer_ocsvm_features.txt")
    with open(feat_path) as f:
        feature_names = [line.strip() for line in f if line.strip()]
    record("3.2a", "PASS", "Feature names loaded",
           f"{len(feature_names)} features from {feat_path}")

    scaler_path = os.path.join(RESULTS_DIR, "transformer_ocsvm_scaler.pkl")
    scaler = joblib.load(scaler_path)
    record("3.2b", "PASS", "Scaler loaded", f"Type: {type(scaler).__name__}")

    df_day, features = load_processed(test_file, TIME_COL, LOB_COLUMNS)
    for col in feature_names:
        if col not in features.columns:
            features[col] = 0.0
    features = features[feature_names]

    scaled = scaler.transform(features.values.astype(np.float32)).astype(np.float32)
    has_nan_scaled = np.isnan(scaled).any()
    has_inf_scaled = np.isinf(scaled).any()
    if has_nan_scaled or has_inf_scaled:
        nan_pct = 100 * np.isnan(scaled).mean()
        record("3.2c", "WARNING", "Scaled data quality",
               f"NaN: {nan_pct:.1f}%, Inf: {np.isinf(scaled).any()}")
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        record("3.2c", "PASS", "Scaled data quality", "No NaN or Inf")

    sequences = create_sequences(scaled, SEQ_LENGTH)
    n_seqs = sequences.shape[0]
    record("3.2d", "PASS", "Sequences created",
           f"Shape: {sequences.shape}")

    num_features = len(feature_names)
    try:
        weights_path = os.path.join(RESULTS_DIR, "transformer_ocsvm_weights.pth")
        sd = torch.load(weights_path, map_location=DEVICE, weights_only=True)
        model_dim = sd["encoder.embedding.weight"].shape[0]
        num_layers = max(
            int(k.split(".")[3]) for k in sd
            if k.startswith("encoder.transformer_encoder.layers.")
        ) + 1
        dim_feedforward = sd["encoder.transformer_encoder.layers.0.linear1.weight"].shape[0]
        in_proj = sd["encoder.transformer_encoder.layers.0.self_attn.in_proj_weight"].shape[0]
        num_heads = in_proj // (3 * model_dim)
        representation_dim = sd["encoder.bottleneck.weight"].shape[0]
        inferred_cfg = dict(
            model_dim=model_dim, num_heads=num_heads, num_layers=num_layers,
            representation_dim=representation_dim, dim_feedforward=dim_feedforward)
        model, ocsvm_obj = load_model(
            "transformer_ocsvm", num_features, weights_path, DEVICE,
            seq_length=SEQ_LENGTH, transformer_cfg=inferred_cfg)
        if isinstance(ocsvm_obj, dict):
            ocsvm_inst = OCSVM(nu=0.01, n_components=300, sgd_lr=0.01, sgd_epochs=500)
            ocsvm_inst.load_state_dict(ocsvm_obj)
            ocsvm_obj = ocsvm_inst

        record("3.2e", "PASS", "TF-OCSVM model loaded",
               f"Input dim={num_features}, seq_len={SEQ_LENGTH}, "
               f"inferred cfg={inferred_cfg}")
    except Exception as e:
        record("3.2e", "FAIL", "TF-OCSVM model loading", str(e))
        return

    try:
        batch_size = min(64, n_seqs)
        x_batch = torch.tensor(sequences[:batch_size], dtype=torch.float32)
        with torch.no_grad():
            latent = model.get_representation(x_batch)
        record("3.2f", "PASS", "TF encoder forward pass",
               f"Latent shape: {latent.shape}")

        scores = ocsvm_obj.dissimilarity_score(latent.cpu().numpy())
        if np.all(np.isfinite(scores)):
            record("3.3a", "PASS", "Dissimilarity scores on real data",
                   f"Range: [{scores.min():.4f}, {scores.max():.4f}], "
                   f"mean={scores.mean():.4f}, std={scores.std():.4f}")
        else:
            n_bad = np.sum(~np.isfinite(scores))
            record("3.3a", "FAIL", "Dissimilarity scores",
                   f"{n_bad}/{len(scores)} non-finite")

        tau = OCSVM.fit_baseline_tau(scores, contamination=0.01)
        record("3.3b", "PASS", "Baseline tau on real data",
               f"tau={tau:.6f}")

        preds = ocsvm_obj.predict(latent.cpu().numpy(), tau=tau)
        n_flagged = (preds == 1).sum()
        record("3.3c", "PASS", "Predictions on real data",
               f"{n_flagged}/{len(preds)} flagged ({100*n_flagged/len(preds):.1f}%)")

    except Exception as e:
        record("3.2f", "FAIL", "TF encoder forward pass", traceback.format_exc())
        return

    try:
        n_check = min(2000, n_seqs)
        all_scores = []
        x_all = torch.tensor(sequences[:n_check], dtype=torch.float32)
        with torch.no_grad():
            for i in range(0, n_check, 256):
                batch = x_all[i:i+256]
                lat = model.get_representation(batch)
                sc = ocsvm_obj.dissimilarity_score(lat.cpu().numpy())
                all_scores.append(sc)
        all_scores = np.concatenate(all_scores)
        if np.all(np.isfinite(all_scores)):
            record("3.3d", "PASS", "Batch inference (2000 seqs)",
                   f"{len(all_scores)} scores, all finite, "
                   f"range=[{all_scores.min():.4f}, {all_scores.max():.4f}]")
        else:
            record("3.3d", "FAIL", "Batch inference",
                   f"{np.sum(~np.isfinite(all_scores))} non-finite scores")
    except Exception as e:
        record("3.3d", "FAIL", "Batch inference", traceback.format_exc())
        return

    from detection.thresholds.pot import PeakOverThreshold
    from detection.thresholds.dspot import DriftStreamingPeakOverThreshold
    from detection.thresholds.rfdr import RollingFalseDiscoveryRate

    try:
        z_pot, t_pot = PeakOverThreshold(all_scores, num_candidates=10,
                                          risk=0.001, init_level=0.98)
        record("3.3e", "PASS", "POT on real scores",
               f"z={z_pot:.4f}, init_t={t_pot:.4f}")
    except Exception as e:
        record("3.3e", "FAIL", "POT on real scores", str(e))

    try:
        z_dspot = DriftStreamingPeakOverThreshold(
            all_scores, num_init=500, depth=50, num_candidates=10,
            risk=0.001, init_level=0.98)
        z_dspot = np.array(z_dspot)
        record("3.3f", "PASS", "DSPOT on real scores",
               f"Threshold range: [{z_dspot.min():.4f}, {z_dspot.max():.4f}]")
    except Exception as e:
        record("3.3f", "FAIL", "DSPOT on real scores", str(e))

    try:
        rfdr = RollingFalseDiscoveryRate(window_size=500, alpha=0.05)
        rfdr_dets = []
        for s in all_scores:
            is_anom, _ = rfdr.process_new_score(float(s))
            rfdr_dets.append(int(is_anom))
        rfdr_dets = np.array(rfdr_dets)
        record("3.3g", "PASS", "RFDR on real scores",
               f"{rfdr_dets.sum()} detections ({100*rfdr_dets.mean():.1f}%)")
    except Exception as e:
        record("3.3g", "FAIL", "RFDR on real scores", str(e))


if __name__ == "__main__":
    print("=" * 60)
    print("LAYER 3: INTEGRATION CHECK")
    print("=" * 60)

    test_file = check_3_1()
    if test_file is not None:
        check_3_2_3(test_file)

    print(f"\n{'='*60}")
    print("LAYER 3 SUMMARY")
    print(f"{'='*60}")
    n_pass = sum(1 for r in RESULTS if r[1] == "PASS")
    n_fail = sum(1 for r in RESULTS if r[1] == "FAIL")
    n_warn = sum(1 for r in RESULTS if r[1] == "WARNING")
    print(f"PASS: {n_pass}  |  FAIL: {n_fail}  |  WARNING: {n_warn}")
    if n_fail > 0:
        print("\nFAILED checks:")
        for r in RESULTS:
            if r[1] == "FAIL":
                print(f"  {r[0]}: {r[2]} -- {r[3]}")
    if n_warn > 0:
        print("\nWARNINGS:")
        for r in RESULTS:
            if r[1] == "WARNING":
                print(f"  {r[0]}: {r[2]} -- {r[3]}")
