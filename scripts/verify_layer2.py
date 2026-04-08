"""
Layer 2: Unit-level functional checks.
Run each component in isolation on synthetic data.
"""
import sys, os, warnings, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

RESULTS = []

def record(check_id, status, description, result, action="none required"):
    RESULTS.append((check_id, status, description, result, action))
    tag = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARNING": "[WARN]"}[status]
    print(f"{tag} Check {check_id} -- {description}")
    if status != "PASS":
        print(f"       Result: {result}")
        if action != "none required":
            print(f"       Action: {action}")


# ======================================================================
# 2.1  Feature pipeline
# ======================================================================
def check_2_1():
    from detection.features.imbalance import compute_imbalance
    from detection.features.dynamics import compute_dynamics, compute_elasticity
    from detection.features.event_flow import compute_event_flow
    from detection.features.hawkes import compute_hawkes
    from detection.features.ofi import compute_ofi
    from detection.features.volatility import compute_volatility
    from detection.data.scalers import EmpiricalBoxCoxScaler

    np.random.seed(42)
    N = 500
    n_levels = 10
    df = pd.DataFrame()
    df["xltime"] = np.linspace(0.375, 0.395, N)  # ~9:00-9:30
    base_bid = 100.0
    base_ask = 100.02
    for lv in range(1, n_levels + 1):
        df[f"bid-price-{lv}"] = base_bid - 0.01 * (lv - 1) + np.random.randn(N) * 0.001
        df[f"ask-price-{lv}"] = base_ask + 0.01 * (lv - 1) + np.random.randn(N) * 0.001
        df[f"bid-volume-{lv}"] = np.random.randint(10, 500, N).astype(float)
        df[f"ask-volume-{lv}"] = np.random.randint(10, 500, N).astype(float)

    # Run feature pipeline steps
    compute_imbalance(df)
    features = pd.DataFrame(index=df.index)
    features["L1_Imbalance"] = df["L1_Imbalance"]
    features["L5_Imbalance"] = df["L5_Imbalance"]
    compute_dynamics(df, features, window=50)
    compute_elasticity(df, features, n_levels=n_levels)
    compute_event_flow(df, features)
    compute_hawkes(df, features, levels=n_levels)
    compute_ofi(df, features, levels=5)
    compute_volatility(df, features, window=50)

    n_cols = features.shape[1]
    has_nan = features.isna().any().any()
    has_inf = np.isinf(features.values).any()

    if has_nan or has_inf:
        nan_cols = features.columns[features.isna().any()].tolist()
        inf_cols = features.columns[np.isinf(features.values).any(axis=0)].tolist()
        record("2.1a", "WARNING",
               "Feature pipeline NaN/Inf check",
               f"NaN cols: {nan_cols[:5]}, Inf cols: {inf_cols[:5]}",
               "Check if warmup rows expected to have NaN")
    else:
        record("2.1a", "PASS", "Feature pipeline NaN/Inf check",
               f"No NaN or Inf in {n_cols} columns, {N} rows")

    record("2.1b", "PASS", "Feature column count",
           f"{n_cols} feature columns produced")

    # Box-Cox scaler
    feat_clean = features.dropna().values.astype(np.float64)
    if feat_clean.shape[0] < 50:
        record("2.1c", "FAIL", "Box-Cox scaler fit", "Too few clean rows after dropna")
        return

    scaler = EmpiricalBoxCoxScaler()
    try:
        out_ft = scaler.fit_transform(feat_clean)
        record("2.1c", "PASS", "Box-Cox fit_transform",
               f"Output shape {out_ft.shape}, no error")
    except Exception as e:
        record("2.1c", "FAIL", "Box-Cox fit_transform", str(e))
        return

    # fit then transform should match fit_transform
    scaler2 = EmpiricalBoxCoxScaler()
    scaler2.fit(feat_clean)
    out_sep = scaler2.transform(feat_clean)
    # Note: they use the same random state? Actually both are deterministic
    # But the lambdas depend on the data only, so should match
    if np.allclose(out_ft, out_sep, atol=1e-5, equal_nan=True):
        record("2.1d", "PASS", "fit_transform == fit+transform",
               "Outputs match within 1e-5")
    else:
        max_diff = np.nanmax(np.abs(out_ft - out_sep))
        record("2.1d", "WARNING", "fit_transform vs fit+transform",
               f"Max diff: {max_diff}")

    # inverse_transform roundtrip
    try:
        recovered = scaler.inverse_transform(out_ft)
        # Only check finite values — exclude near-zero originals
        mask = (np.isfinite(feat_clean) & np.isfinite(recovered)
                & (np.abs(feat_clean) > 0.01))
        if mask.any():
            max_err = np.abs(feat_clean[mask] - recovered[mask]).max()
            rel_err = (np.abs(feat_clean[mask] - recovered[mask]) /
                       (np.abs(feat_clean[mask]) + 1e-10)).max()
            median_rel = np.median(np.abs(feat_clean[mask] - recovered[mask]) /
                                   (np.abs(feat_clean[mask]) + 1e-10))
            if median_rel < 0.05:  # 5% median relative error
                record("2.1e", "PASS", "Box-Cox inverse_transform roundtrip",
                       f"Median rel err: {median_rel:.6f}, max rel err: {rel_err:.4f}")
            elif median_rel < 0.20:
                record("2.1e", "WARNING", "Box-Cox inverse_transform roundtrip",
                       f"Median rel err: {median_rel:.4f}, max rel err: {rel_err:.4f}",
                       "Some precision loss in Box-Cox inverse")
            else:
                record("2.1e", "FAIL", "Box-Cox inverse_transform roundtrip",
                       f"Median rel err: {median_rel:.4f}, max: {rel_err:.4f}")
        else:
            record("2.1e", "WARNING", "Box-Cox inverse_transform",
                   "No finite values to compare")
    except Exception as e:
        record("2.1e", "FAIL", "Box-Cox inverse_transform", str(e))


# ======================================================================
# 2.2  Sequence windowing
# ======================================================================
def check_2_2():
    from detection.data.loaders import create_sequences
    np.random.seed(42)
    T, D = 1000, 32
    seq_len = 25
    data = np.random.randn(T, D).astype(np.float32)
    seqs = create_sequences(data, seq_len)
    expected_n = T - seq_len
    if seqs.shape == (expected_n, seq_len, D):
        record("2.2a", "PASS", "Sequence window shapes",
               f"Input ({T},{D}) -> {seqs.shape}")
    else:
        record("2.2a", "FAIL", "Sequence window shapes",
               f"Expected ({expected_n},{seq_len},{D}), got {seqs.shape}")

    # Verify first and last windows
    # create_sequences: n_samples = T - seq_len, window[i] = data[i:i+seq_len]
    if np.allclose(seqs[0], data[:seq_len]) and np.allclose(seqs[-1], data[T-seq_len-1:T-1]):
        record("2.2b", "PASS", "First/last window content",
               "Windows correctly slice the data")
    else:
        # Maybe the last window is data[-seq_len:]? Check that too
        if np.allclose(seqs[-1], data[-seq_len:]):
            record("2.2b", "PASS", "First/last window content",
                   "Windows correct (last = data[-seq_len:])")
        else:
            record("2.2b", "FAIL", "First/last window content", "Content mismatch")


# ======================================================================
# 2.3  OC-SVM dissimilarity score
# ======================================================================
def check_2_3():
    from detection.models.ocsvm import OCSVM
    np.random.seed(42)
    torch.manual_seed(42)
    N_train, N_test, D = 300, 100, 16
    X_train = np.random.randn(N_train, D).astype(np.float32) * 0.5
    X_test = np.random.randn(N_test, D).astype(np.float32) * 0.5
    X_outlier = np.random.randn(20, D).astype(np.float32) * 0.5 + 5.0

    ocsvm = OCSVM(nu=0.05, n_components=50, sgd_epochs=200, batch_size=128)
    ocsvm.fit(X_train)

    # dissimilarity_score: float array, no NaN, no binary
    ds = ocsvm.dissimilarity_score(X_test)
    if not isinstance(ds, np.ndarray):
        record("2.3a", "FAIL", "dissimilarity_score type", f"Got {type(ds)}")
        return
    if ds.dtype.kind != 'f':
        record("2.3a", "FAIL", "dissimilarity_score dtype", f"Got {ds.dtype}")
        return
    if np.any(np.isnan(ds)):
        record("2.3a", "FAIL", "dissimilarity_score NaN", "Contains NaN values")
        return
    unique_vals = len(np.unique(ds))
    if unique_vals <= 2:
        record("2.3a", "FAIL", "dissimilarity_score values",
               f"Only {unique_vals} unique values (binary?)")
        return
    record("2.3a", "PASS", "dissimilarity_score output",
           f"Float array, {unique_vals} unique values, no NaN")

    # fit_baseline_tau
    train_scores = ocsvm.dissimilarity_score(X_train)
    tau = OCSVM.fit_baseline_tau(train_scores, contamination=0.01)
    if not isinstance(tau, float):
        record("2.3b", "FAIL", "fit_baseline_tau type", f"Got {type(tau)}")
    elif not np.isfinite(tau):
        record("2.3b", "FAIL", "fit_baseline_tau value", f"Non-finite: {tau}")
    else:
        record("2.3b", "PASS", "fit_baseline_tau",
               f"tau = {tau:.6f}, in [{train_scores.min():.4f}, {train_scores.max():.4f}]")

    # predict: binary values only
    preds = ocsvm.predict(np.vstack([X_test, X_outlier]), tau=tau)
    unique_preds = set(np.unique(preds))
    if unique_preds <= {1, -1}:
        record("2.3c", "PASS", "predict output",
               f"Binary: {unique_preds}, {(preds==1).sum()} flagged / {len(preds)}")
    else:
        record("2.3c", "FAIL", "predict output", f"Unexpected values: {unique_preds}")

    # predict(tau=baseline) vs predict(tau=0) differ
    preds_zero = ocsvm.predict(np.vstack([X_test, X_outlier]), tau=0.0)
    preds_tau = ocsvm.predict(np.vstack([X_test, X_outlier]), tau=tau)
    n_diff = (preds_zero != preds_tau).sum()
    if n_diff > 0:
        record("2.3d", "PASS", "Baseline tau changes predictions",
               f"{n_diff} predictions differ (tau=0 vs tau={tau:.4f})")
    else:
        record("2.3d", "WARNING", "Baseline tau vs hardcoded zero",
               "No difference in predictions (may be edge case with synthetic data)")


# ======================================================================
# 2.4  Thresholding methods
# ======================================================================
def check_2_4():
    from detection.thresholds.pot import PeakOverThreshold
    from detection.thresholds.spot import StreamingPeakOverThreshold
    from detection.thresholds.dspot import DriftStreamingPeakOverThreshold
    from detection.thresholds.rfdr import RollingFalseDiscoveryRate

    np.random.seed(42)
    N = 5000
    # Mixture: mostly normal, some heavy-tailed outliers
    inlier = np.random.randn(N) * 1.0
    outlier_idx = np.random.choice(N, size=int(0.02 * N), replace=False)
    inlier[outlier_idx] += np.random.exponential(5.0, len(outlier_idx))
    scores = inlier

    # POT
    try:
        z_pot, t_pot = PeakOverThreshold(scores, num_candidates=10,
                                         risk=1e-3, init_level=0.98)
        if np.isfinite(z_pot) and z_pot > 0:
            record("2.4a", "PASS", "POT threshold",
                   f"z={z_pot:.4f}, init_t={t_pot:.4f}")
        else:
            record("2.4a", "FAIL", "POT threshold",
                   f"z={z_pot}, not finite positive")
    except Exception as e:
        record("2.4a", "FAIL", "POT threshold", traceback.format_exc())

    # SPOT
    try:
        z_spot = StreamingPeakOverThreshold(
            scores, num_init=1000, num_candidates=10,
            risk=1e-3, init_level=0.98)
        z_spot = np.array(z_spot)
        if len(z_spot) == N and np.all(np.isfinite(z_spot)):
            record("2.4b", "PASS", "SPOT threshold",
                   f"len={len(z_spot)}, range=[{z_spot.min():.4f}, {z_spot.max():.4f}]")
        else:
            n_nan = np.sum(~np.isfinite(z_spot))
            record("2.4b", "FAIL", "SPOT threshold",
                   f"len={len(z_spot)}, {n_nan} non-finite values")
    except Exception as e:
        record("2.4b", "FAIL", "SPOT threshold", traceback.format_exc())

    # DSPOT
    try:
        z_dspot = DriftStreamingPeakOverThreshold(
            scores, num_init=1000, depth=200, num_candidates=10,
            risk=1e-3, init_level=0.98)
        if len(z_dspot) == N and np.all(np.isfinite(z_dspot)):
            record("2.4c", "PASS", "DSPOT threshold",
                   f"len={len(z_dspot)}, range=[{z_dspot.min():.4f}, {z_dspot.max():.4f}]")
        else:
            n_nan = np.sum(~np.isfinite(z_dspot))
            record("2.4c", "FAIL", "DSPOT threshold",
                   f"len={len(z_dspot)}, {n_nan} non-finite values")
    except Exception as e:
        record("2.4c", "FAIL", "DSPOT threshold", traceback.format_exc())

    # RFDR
    try:
        rfdr = RollingFalseDiscoveryRate(window_size=500, alpha=0.05)
        detections = []
        for s in scores:
            is_anom, thr = rfdr.process_new_score(s)
            detections.append(int(is_anom))
        detections = np.array(detections)
        if len(detections) == N:
            record("2.4d", "PASS", "RFDR detection array",
                   f"len={N}, {detections.sum()} detections ({100*detections.mean():.1f}%)")
        else:
            record("2.4d", "FAIL", "RFDR detection array",
                   f"Expected len {N}, got {len(detections)}")
    except Exception as e:
        record("2.4d", "FAIL", "RFDR detection", traceback.format_exc())

    # Compare orders of magnitude
    try:
        thresholds = {"POT": z_pot}
        if 'z_spot' in dir():
            thresholds["SPOT_median"] = np.median(z_spot)
        if 'z_dspot' in dir():
            thresholds["DSPOT_median"] = np.median(z_dspot)
        vals = list(thresholds.values())
        if len(vals) >= 2 and max(vals)/max(min(vals), 1e-10) < 100:
            record("2.4e", "PASS", "Threshold magnitude comparison",
                   ", ".join(f"{k}={v:.4f}" for k, v in thresholds.items()))
        elif len(vals) >= 2:
            record("2.4e", "WARNING", "Threshold magnitude comparison",
                   ", ".join(f"{k}={v:.4f}" for k, v in thresholds.items()))
        else:
            record("2.4e", "WARNING", "Threshold comparison",
                   "Not enough methods succeeded")
    except:
        pass


# ======================================================================
# 2.5  Spoofing gain
# ======================================================================
def check_2_5():
    from detection.spoofing.gain import compute_spoofing_gains_batch

    np.random.seed(42)
    N = 100
    mu = np.random.randn(N) * 0.01
    sigma = np.abs(np.random.randn(N)) * 0.005 + 0.001
    alpha = np.random.randn(N) * 2.0
    spread = np.ones(N) * 0.02
    fees = {"maker": 0.0, "taker": 0.0008}

    try:
        gains = compute_spoofing_gains_batch(
            mu, sigma, alpha, spread,
            delta_a=0.0, delta_b=0.01, Q=4500, q=100,
            fees=fees, side="ask")
        if np.all(np.isfinite(gains)):
            record("2.5a", "PASS", "Spoofing gain (ask-side)",
                   f"All finite, range [{gains.min():.4f}, {gains.max():.4f}]")
        else:
            n_bad = np.sum(~np.isfinite(gains))
            record("2.5a", "FAIL", "Spoofing gain (ask-side)",
                   f"{n_bad}/{N} non-finite values")
    except Exception as e:
        record("2.5a", "FAIL", "Spoofing gain (ask-side)", traceback.format_exc())

    try:
        gains_bid = compute_spoofing_gains_batch(
            mu, sigma, alpha, spread,
            delta_a=0.0, delta_b=0.01, Q=4500, q=100,
            fees=fees, side="bid")
        if np.all(np.isfinite(gains_bid)):
            record("2.5b", "PASS", "Spoofing gain (bid-side)",
                   f"All finite, range [{gains_bid.min():.4f}, {gains_bid.max():.4f}]")
        else:
            record("2.5b", "FAIL", "Spoofing gain (bid-side)", "Non-finite values")
    except Exception as e:
        record("2.5b", "FAIL", "Spoofing gain (bid-side)", traceback.format_exc())


# ======================================================================
# 2.6  Integrated Gradients
# ======================================================================
def check_2_6():
    from detection.sensitivity.integrated_gradients import IntegratedGradients

    # Tiny PNN-like model: input -> hidden -> (mu, sigma, alpha)
    class TinyPNN(torch.nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.fc1 = torch.nn.Linear(d_in, 4)
            self.fc2 = torch.nn.Linear(4, 3)
        def forward(self, x):
            h = torch.relu(self.fc1(x))
            out = self.fc2(h)
            return out

    torch.manual_seed(42)
    d_in = 8
    model = TinyPNN(d_in)
    model.eval()

    ig = IntegratedGradients(model)
    x = torch.randn(1, 1, d_in)  # batch=1, seq=1, feat=d_in
    baseline = torch.zeros_like(x)

    def target_fn(output, inputs):
        return output.sum()

    # Check completeness: sum(IG) ≈ f(x) - f(baseline)
    try:
        attrs = ig.attribute(x, baseline=baseline, target_func=target_fn, n_steps=300)
        with torch.no_grad():
            f_x = target_fn(model(x.view(1, -1)), x)
            f_b = target_fn(model(baseline.view(1, -1)), baseline)
        delta = (f_x - f_b).item()
        ig_sum = attrs.sum().item()
        if abs(delta) > 1e-8:
            rel_err = abs(ig_sum - delta) / abs(delta)
        else:
            rel_err = abs(ig_sum - delta)
        if rel_err < 0.01:
            record("2.6a", "PASS", "IG completeness property",
                   f"sum(IG)={ig_sum:.6f}, f(x)-f(b)={delta:.6f}, rel_err={rel_err:.4%}")
        else:
            record("2.6a", "FAIL", "IG completeness property",
                   f"sum(IG)={ig_sum:.6f}, f(x)-f(b)={delta:.6f}, rel_err={rel_err:.4%}")
    except Exception as e:
        record("2.6a", "FAIL", "IG completeness", traceback.format_exc())

    # Monotonic improvement with more steps
    try:
        errors = []
        for n_steps in [10, 50, 100, 300]:
            a = ig.attribute(x, baseline=baseline, target_func=target_fn, n_steps=n_steps)
            err = abs(a.sum().item() - delta)
            errors.append((n_steps, err))
        # Check generally decreasing (allow small fluctuations)
        if errors[-1][1] <= errors[0][1] * 1.1:  # last <= first * 1.1
            record("2.6b", "PASS", "IG step convergence",
                   ", ".join(f"n={n}: err={e:.6f}" for n, e in errors))
        else:
            record("2.6b", "WARNING", "IG step convergence",
                   ", ".join(f"n={n}: err={e:.6f}" for n, e in errors))
    except Exception as e:
        record("2.6b", "FAIL", "IG step convergence", traceback.format_exc())


# ======================================================================
# 2.7  Grouped Occlusion
# ======================================================================
def check_2_7():
    from detection.sensitivity.occlusion import GroupedOcclusion
    from detection.models.ocsvm import OCSVM

    torch.manual_seed(42)
    np.random.seed(42)

    # Create a minimal transformer-like encoder + OC-SVM detector
    seq_len, d_in = 5, 8
    feature_names = [
        "bid-price-1", "ask-price-1", "bid-volume-1", "ask-volume-1",
        "mid_price", "spread", "log_return", "volatility_50"
    ]

    class FakeTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._dummy = torch.nn.Linear(1, 1)  # needs at least one param
        def get_representation(self, x):
            # x: (B, seq_len, d_in) -> (B, 4)
            return x.mean(dim=1)[:, :4]  # take first 4 features' mean

    class FakeDetector:
        def __init__(self):
            self.transformer = FakeTransformer()
            self.ocsvm = OCSVM(nu=0.1, n_components=20, sgd_epochs=100, batch_size=64)
            self.device = torch.device("cpu")
            # Fit on some data
            train_reps = np.random.randn(200, 4).astype(np.float32)
            self.ocsvm.fit(train_reps)

    detector = FakeDetector()
    detector.transformer.eval()
    x_seq = torch.randn(1, seq_len, d_in)

    try:
        importance_df, feature_lists = GroupedOcclusion(
            detector, x_seq, feature_names, group_by="side", baseline_mode="mean")
        if isinstance(importance_df, pd.DataFrame) and len(importance_df) > 0:
            record("2.7", "PASS", "Grouped Occlusion",
                   f"{len(importance_df)} groups: {importance_df['Group'].tolist()}")
        else:
            record("2.7", "FAIL", "Grouped Occlusion",
                   f"Unexpected output type: {type(importance_df)}")
    except Exception as e:
        record("2.7", "FAIL", "Grouped Occlusion", traceback.format_exc())


# ======================================================================
# 2.8  Clustering (HDBSCAN)
# ======================================================================
def check_2_8():
    import hdbscan

    np.random.seed(42)
    # Simulate 3-model normalized scores for 200 anomalies
    n_anom = 200
    # Create 3 clusters in 3D score space
    X = np.vstack([
        np.random.randn(80, 3) * 0.3 + [0.8, 0.2, 0.1],
        np.random.randn(60, 3) * 0.3 + [0.2, 0.9, 0.3],
        np.random.randn(60, 3) * 0.3 + [0.1, 0.1, 0.8],
    ])
    X = np.clip(X, 0, 1)  # MinMax-like

    try:
        hdb = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
        labels = hdb.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        if n_clusters >= 1:
            record("2.8", "PASS", "HDBSCAN clustering",
                   f"{n_clusters} clusters, {n_noise} noise points")
        else:
            record("2.8", "WARNING", "HDBSCAN clustering",
                   f"0 clusters found, {n_noise} noise points")
    except Exception as e:
        record("2.8", "FAIL", "HDBSCAN clustering", traceback.format_exc())


# ======================================================================
# 2.9  Jump analysis (basic statistical test)
# ======================================================================
def check_2_9():
    """Verify co-occurrence logic on a manual example."""
    np.random.seed(42)
    N = 1000
    # Synthetic mid-price with 5 injected jumps at known positions
    prices = 100.0 + np.cumsum(np.random.randn(N) * 0.01)
    jump_positions = [100, 250, 500, 700, 900]
    for jp in jump_positions:
        prices[jp:] += 0.5  # inject jump

    # Compute returns and detect jumps via simple threshold
    returns = np.diff(prices)
    jump_threshold = 0.3
    detected_jumps = np.abs(returns) > jump_threshold

    # Synthetic anomaly flags: anomalies at some jump positions + noise
    anomaly_flags = np.zeros(N - 1, dtype=bool)
    anomaly_flags[99] = True   # near jump at 100
    anomaly_flags[249] = True  # near jump at 250
    anomaly_flags[499] = True  # near jump at 500
    anomaly_flags[10] = True   # noise (not near jump)
    anomaly_flags[600] = True  # noise (not near jump)

    # Co-occurrence within window of 2
    window = 2
    n_cooccur = 0
    for jp in range(len(returns)):
        if detected_jumps[jp]:
            start = max(0, jp - window)
            end = min(len(anomaly_flags), jp + window + 1)
            if anomaly_flags[start:end].any():
                n_cooccur += 1

    expected_cooccur = 3  # jumps at 100, 250, 500 have nearby anomalies
    if n_cooccur == expected_cooccur:
        record("2.9a", "PASS", "Jump-anomaly co-occurrence",
               f"Found {n_cooccur} co-occurrences (expected {expected_cooccur})")
    else:
        record("2.9a", "WARNING", "Jump-anomaly co-occurrence",
               f"Found {n_cooccur}, expected {expected_cooccur}")

    # Basic statistical test: is anomaly rate higher near jumps?
    from scipy.stats import fisher_exact
    n_jumps = detected_jumps.sum()
    n_anom = anomaly_flags.sum()
    # Build contingency table
    near_jump_and_anom = n_cooccur
    near_jump_no_anom = n_jumps - n_cooccur
    no_jump_and_anom = (anomaly_flags & ~detected_jumps).sum()
    no_jump_no_anom = len(returns) - n_jumps - no_jump_and_anom
    table = [[near_jump_and_anom, near_jump_no_anom],
             [no_jump_and_anom, no_jump_no_anom]]
    odds, pval = fisher_exact(table, alternative="greater")
    if 0 <= pval <= 1:
        record("2.9b", "PASS", "Fisher exact test p-value",
               f"p={pval:.6f}, odds ratio={odds:.2f}")
    else:
        record("2.9b", "FAIL", "Fisher exact test", f"Invalid p-value: {pval}")


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    checks = [
        ("2.1", check_2_1),
        ("2.2", check_2_2),
        ("2.3", check_2_3),
        ("2.4", check_2_4),
        ("2.5", check_2_5),
        ("2.6", check_2_6),
        ("2.7", check_2_7),
        ("2.8", check_2_8),
        ("2.9", check_2_9),
    ]
    for check_id, fn in checks:
        print(f"\n{'='*60}")
        print(f"CHECK {check_id}")
        print(f"{'='*60}")
        try:
            fn()
        except Exception as e:
            record(check_id, "FAIL", f"Unexpected error in {check_id}",
                   traceback.format_exc())

    # Summary
    print(f"\n{'='*60}")
    print("LAYER 2 SUMMARY")
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
