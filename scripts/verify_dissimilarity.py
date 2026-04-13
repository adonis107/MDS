"""End-to-end verification of the dissimilarity score and baseline tau.

Generates synthetic data, fits an OC-SVM, and checks:
1. dissimilarity_score == -decision_function  (by definition)
2. predict(X, tau=0) matches sign of dissimilarity_score
3. fit_baseline_tau returns the correct quantile
4. Baseline tau lies between min and max dissimilarity scores
5. Anomalies (outlier cluster) have higher dissimilarity than normals
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from detection.models.ocsvm import OCSVM


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    n_normal = 500
    n_outlier = 20
    d = 16

    X_normal = np.random.randn(n_normal, d).astype(np.float32) * 0.5
    X_outlier = np.random.randn(n_outlier, d).astype(np.float32) * 0.5 + 5.0
    X_all = np.vstack([X_normal, X_outlier])
    labels = np.array([0] * n_normal + [1] * n_outlier)

    ocsvm = OCSVM(nu=0.05, n_components=50, sgd_epochs=200, batch_size=128)
    ocsvm.fit(X_normal)

    df = ocsvm.decision_function(X_all)
    ds = ocsvm.dissimilarity_score(X_all)
    assert np.allclose(ds, -df, atol=1e-6), "FAIL: dissimilarity != -decision_function"
    print("[PASS] Check 1: dissimilarity_score == -decision_function")

    preds = ocsvm.predict(X_all, tau=0.0)
    expected = np.where(ds >= 0.0, 1, -1)
    assert np.array_equal(preds, expected), "FAIL: predict(tau=0) mismatch"
    print("[PASS] Check 2: predict(X, tau=0) consistent with dissimilarity >= 0")

    scores_train = ocsvm.dissimilarity_score(X_normal)
    contamination = 0.01
    tau = OCSVM.fit_baseline_tau(scores_train, contamination=contamination)
    expected_tau = float(np.quantile(scores_train.astype(np.float64), 1.0 - contamination))
    assert abs(tau - expected_tau) < 1e-10, f"FAIL: tau={tau} != expected={expected_tau}"
    print(f"[PASS] Check 3: fit_baseline_tau = {tau:.6f} (quantile {1-contamination:.2f})")

    assert scores_train.min() <= tau <= scores_train.max(), "FAIL: tau outside score range"
    print(f"[PASS] Check 4: tau in [{scores_train.min():.4f}, {scores_train.max():.4f}]")

    mean_normal = ds[:n_normal].mean()
    mean_outlier = ds[n_normal:].mean()
    assert mean_outlier > mean_normal, (
        f"FAIL: mean outlier dissimilarity ({mean_outlier:.4f}) <= "
        f"mean normal ({mean_normal:.4f})"
    )
    print(f"[PASS] Check 5: mean dissimilarity: normal={mean_normal:.4f}, "
          f"outlier={mean_outlier:.4f}")

    preds_baseline = ocsvm.predict(X_all, tau=tau)
    n_flagged_zero = (preds == 1).sum()
    n_flagged_tau = (preds_baseline == 1).sum()
    assert n_flagged_tau <= n_flagged_zero, "FAIL: baseline tau flags more than tau=0"
    print(f"[PASS] Check 6: flagged with tau=0: {n_flagged_zero}, "
          f"with baseline tau: {n_flagged_tau}")

    print("\n=== All 6 checks passed ===")
    print(f"  OC-SVM fitted on {n_normal} normal points ({d}-dim)")
    print(f"  Tested on {n_normal} normal + {n_outlier} outliers")
    print(f"  Baseline tau (contamination={contamination}): {tau:.6f}")
    print(f"  Decision boundary at tau=0 flags {n_flagged_zero}/{len(X_all)} events")
    print(f"  Baseline tau flags {n_flagged_tau}/{len(X_all)} events")


if __name__ == "__main__":
    main()
