"""
Generate all diagnostic figures for Section 5.1 (Model Diagnostics).

Outputs PDF figures to figures/diagnostics/ for both training years (2015, 2017).
Run from repo root:  python scripts/generate_diagnostics_figures.py
"""

import os, sys, glob, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import joblib
from scipy.special import erf, owens_t

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# fast skew-normal CDF using Owen's T (1000x vs scipy)
def _fast_skewnorm_cdf(x, mu, sigma, alpha):
    """F(x; mu, sigma, alpha) = Φ(z) - 2·T(z, alpha), z=(x-mu)/sigma."""
    z = (np.asarray(x, dtype=np.float64) - mu) / sigma
    phi_z = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    return phi_z - 2.0 * owens_t(z, np.asarray(alpha, dtype=np.float64))

import detection.spoofing.gain as _gain_mod
_gain_mod.skewed_gaussian_cdf = _fast_skewnorm_cdf

from detection.data.loaders import create_sequences, load_processed
from detection.data.preprocessing import split_first_hour_blocks
from detection.models.transformer import BottleneckTransformer
from detection.models.ocsvm import OCSVM
from detection.models.pnn import PNN, SkewedGaussianNLL
from detection.models.prae import PRAE
from sklearn.decomposition import PCA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_DIR = os.path.join("data", "processed", "TOTF.PA-book")
OUT_DIR = os.path.join("figures", "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LENGTH = 25
BATCH_SIZE = 512
LOB_COLUMNS = [
    f"{side}-{typ}-{lvl}"
    for lvl in range(1, 11)
    for side, typ in [("bid","price"),("bid","volume"),("ask","price"),("ask","volume")]
]

FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
TEST_FILES = [FILES[22], FILES[23], FILES[24], FILES[25], FILES[26]]

# Plot style
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
BLUE = "#4878CF"
ORANGE = "#E8884A"
RED = "#C44E52"
GREEN = "#6AB187"

NUM_HOLDOUT = 12
NUM_TAU_DAYS = 3
CONTAMINATION = 0.01


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  Saved: {path}")


def load_features_and_scaler(results_dir, model_prefix):
    feat_path = os.path.join(results_dir, f"{model_prefix}_features.txt")
    with open(feat_path) as f:
        feat_names = [line.strip() for line in f if line.strip()]
    scaler = joblib.load(os.path.join(results_dir, f"{model_prefix}_scaler.pkl"))
    return feat_names, scaler


def prepare_sequences(file_path, feat_names, scaler):
    df, features = load_processed(file_path, "xltime", LOB_COLUMNS)
    feat_df = features.copy()
    for col in feat_names:
        if col not in feat_df.columns:
            feat_df[col] = 0.0
    feat_df = feat_df[feat_names]
    scaled = scaler.transform(feat_df.values.astype(np.float32)).astype(np.float32)
    seqs = create_sequences(scaled, SEQ_LENGTH)
    return df, features, seqs


# ==============================================================================
# TF-OC-SVM Diagnostics
# ==============================================================================
def generate_tfocsvm_diagnostics(year):
    print(f"\n{'='*60}")
    print(f"TF-OC-SVM Diagnostics — {year}")
    print(f"{'='*60}")
    results_dir = os.path.join("results", str(year))

    feat_names, scaler = load_features_and_scaler(results_dir, "transformer_ocsvm")
    num_features = len(feat_names)

    # Load models
    transformer = BottleneckTransformer(
        num_features=num_features, model_dim=128, num_heads=8,
        num_layers=6, representation_dim=128, sequence_length=SEQ_LENGTH,
        dim_feedforward=512,
    )
    transformer.load_state_dict(torch.load(
        os.path.join(results_dir, "transformer_ocsvm_weights.pth"),
        map_location=DEVICE, weights_only=True))
    transformer.eval().to(DEVICE)

    ocsvm = torch.load(
        os.path.join(results_dir, "transformer_ocsvm_detector.pth"),
        map_location=DEVICE, weights_only=False)

    # Encode first test day
    _, _, seqs = prepare_sequences(TEST_FILES[0], feat_names, scaler)
    all_z = []
    with torch.no_grad():
        for start in range(0, len(seqs), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(seqs))
            x_t = torch.tensor(seqs[start:end], dtype=torch.float32).to(DEVICE)
            z = transformer.get_representation(x_t)
            all_z.append(z.cpu())
    z_all = torch.cat(all_z, dim=0)
    z_np = z_all.numpy()
    print(f"  Encoded {z_all.shape[0]} sequences, latent dim {z_all.shape[1]}")

    # 1. Latent space statistics
    dim_var = z_np.var(axis=0)
    dim_mean = z_np.mean(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    axes[0].hist(z_np.flatten(), bins=100, color=BLUE, edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Latent value")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of all latent values")

    axes[1].bar(range(len(dim_var)), np.sort(dim_var)[::-1], color=BLUE, width=1.0)
    axes[1].set_xlabel("Dimension (sorted)")
    axes[1].set_ylabel("Variance")
    axes[1].set_title("Per-dimension variance (sorted)")
    axes[1].set_yscale("log")

    axes[2].bar(range(len(dim_mean)), dim_mean[np.argsort(-dim_var)], color=ORANGE, width=1.0)
    axes[2].set_xlabel("Dimension (sorted by variance)")
    axes[2].set_ylabel("Mean")
    axes[2].set_title("Per-dimension mean (sorted by variance)")
    plt.tight_layout()
    save_fig(fig, f"tfocsvm_latent_statistics_{year}.pdf")

    # 2. Kernel calibration
    gamma = ocsvm._gamma
    n_sample = min(1000, z_all.shape[0])
    idx = np.random.choice(z_all.shape[0], n_sample, replace=False)
    z_sample = z_all[idx]
    z_sq = (z_sample ** 2).sum(dim=1, keepdim=True)
    dist2 = z_sq + z_sq.T - 2.0 * z_sample @ z_sample.T
    dist2 = dist2.clamp(min=0.0)
    triu_idx = torch.triu_indices(n_sample, n_sample, offset=1)
    dist2_pairs = dist2[triu_idx[0], triu_idx[1]].numpy()
    kernel_vals = np.exp(-gamma * dist2_pairs)
    d = z_all.shape[1]
    var_z = z_all.var().item()
    expected_dist2 = 2 * d * var_z
    typical_kernel = np.exp(-gamma * expected_dist2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].hist(kernel_vals, bins=100, color=BLUE, edgecolor="white", alpha=0.85, density=True)
    axes[0].axvline(typical_kernel, color=RED, linestyle="--", label=f"Typical = {typical_kernel:.2e}")
    axes[0].set_xlabel(r"$K(z_i, z_j)$")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Pairwise RBF kernel value distribution")
    axes[0].legend()

    axes[1].hist(dist2_pairs, bins=100, color=ORANGE, edgecolor="white", alpha=0.85, density=True)
    axes[1].axvline(expected_dist2, color=RED, linestyle="--", label=f"E[d²] = {expected_dist2:.1f}")
    axes[1].axvline(np.median(dist2_pairs), color=GREEN, linestyle="--", label=f"Median = {np.median(dist2_pairs):.1f}")
    axes[1].set_xlabel(r"$\|z_i - z_j\|^2$")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Pairwise squared distance distribution")
    axes[1].legend()
    plt.tight_layout()
    save_fig(fig, f"tfocsvm_kernel_calibration_{year}.pdf")

    # 3. Nyström eigenvalue spectrum
    landmarks = ocsvm._landmarks
    K_CC = ocsvm._rbf_kernel(landmarks, landmarks)
    K_CC_np = K_CC.cpu().numpy()
    eigvals = np.linalg.eigvalsh(K_CC_np)
    eigvals = np.sort(eigvals)[::-1]
    eigvals_pos = eigvals[eigvals > 0]
    cumvar = np.cumsum(eigvals_pos) / np.sum(eigvals_pos)
    effective_rank_95 = int(np.searchsorted(cumvar, 0.95) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].semilogy(eigvals_pos, "o-", markersize=1.5, color=BLUE)
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Eigenvalue (log scale)")
    axes[0].set_title(f"$K_{{CC}}$ eigenvalue spectrum ($m={landmarks.shape[0]}$)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(cumvar, color=ORANGE, linewidth=2)
    axes[1].axhline(0.95, color="gray", linestyle="--", alpha=0.7, label="95%")
    axes[1].axhline(0.99, color="gray", linestyle=":", alpha=0.7, label="99%")
    axes[1].axvline(effective_rank_95, color=BLUE, linestyle="--", alpha=0.7,
                    label=f"Eff. rank 95% = {effective_rank_95}")
    axes[1].set_xlabel("Number of eigenvalues")
    axes[1].set_ylabel("Cumulative variance ratio")
    axes[1].set_title("Cumulative variance explained")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, f"tfocsvm_nystrom_spectrum_{year}.pdf")

    # 4. Decision function & dissimilarity
    decision_vals = ocsvm.decision_function(z_np)
    dissimilarity = -decision_vals
    n_outliers = int((decision_vals < 0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].hist(decision_vals, bins=150, color=BLUE, edgecolor="white", alpha=0.85, density=True)
    axes[0].axvline(0, color=RED, linestyle="--", linewidth=1.5, label="Decision boundary ($f=0$)")
    axes[0].set_xlabel(r"$f(z) = w^\top \tilde{\Phi}(z) - \rho$")
    axes[0].set_ylabel("Density")
    axes[0].set_title("OC-SVM decision function")
    axes[0].legend()

    axes[1].hist(dissimilarity, bins=150, color=ORANGE, edgecolor="white", alpha=0.85, density=True)
    axes[1].axvline(0, color=RED, linestyle="--", linewidth=1.5, label=r"Boundary ($\tau=0$)")
    axes[1].set_xlabel(r"Dissimilarity score $-f(z)$")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Dissimilarity score distribution")
    axes[1].legend()
    plt.tight_layout()
    save_fig(fig, f"tfocsvm_decision_function_{year}.pdf")

    # 5. Reconstruction error vs dissimilarity
    rec_errors = []
    with torch.no_grad():
        for start in range(0, len(seqs), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(seqs))
            x_t = torch.tensor(seqs[start:end], dtype=torch.float32).to(DEVICE)
            x_hat = transformer(x_t)
            mse = ((x_hat - x_t) ** 2).mean(dim=(1, 2)).cpu().numpy()
            rec_errors.append(mse)
    rec_errors = np.concatenate(rec_errors)
    corr = float(np.corrcoef(rec_errors, dissimilarity)[0, 1])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(rec_errors, dissimilarity, alpha=0.1, s=2, color=BLUE)
    ax.set_xlabel(r"Reconstruction error $\|x - \hat{x}\|^2$")
    ax.set_ylabel(r"Dissimilarity score $-f(z)$")
    ax.set_title(f"Reconstruction error vs. dissimilarity ($r = {corr:.3f}$)")
    ax.axhline(0, color=RED, linestyle="--", alpha=0.5, label="OC-SVM boundary")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, f"tfocsvm_rec_vs_dissimilarity_{year}.pdf")

    # 6. Latent PCA
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_np)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc = axes[0].scatter(z_2d[:, 0], z_2d[:, 1], c=dissimilarity, cmap="coolwarm",
                         s=1, alpha=0.4, vmin=np.percentile(dissimilarity, 1),
                         vmax=np.percentile(dissimilarity, 99))
    plt.colorbar(sc, ax=axes[0], label="Dissimilarity")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[0].set_title("Latent space: coloured by dissimilarity")

    is_outlier = decision_vals < 0
    axes[1].scatter(z_2d[~is_outlier, 0], z_2d[~is_outlier, 1], s=1, alpha=0.3,
                    color=BLUE, label=f"Inliers ({(~is_outlier).sum():,})")
    if is_outlier.sum() > 0:
        axes[1].scatter(z_2d[is_outlier, 0], z_2d[is_outlier, 1], s=8, alpha=0.7,
                        color=RED, label=f"Outliers ({is_outlier.sum():,})", zorder=5)
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[1].set_title("Latent space: inliers vs. outliers")
    axes[1].legend(markerscale=3)
    plt.tight_layout()
    save_fig(fig, f"tfocsvm_latent_pca_{year}.pdf")

    # 7. Baseline τ on training data
    year_files = sorted(glob.glob(os.path.join(DATA_DIR, f"{year}*.parquet")))
    train_files = year_files[:len(year_files) - NUM_HOLDOUT]

    train_scores_all = []
    with torch.no_grad():
        for tf in train_files[:NUM_TAU_DAYS]:
            try:
                df_day, feat_day = load_processed(tf, "xltime", LOB_COLUMNS)
                for c in feat_names:
                    if c not in feat_day.columns:
                        feat_day[c] = 0.0
                feat_day = feat_day[feat_names]
                train_block, _ = split_first_hour_blocks(
                    df_day["xltime"].values, feat_day, 9.0, 60, 5, 5)
                if len(train_block) < SEQ_LENGTH + 1:
                    continue
                scaled_block = scaler.transform(
                    train_block.values.astype(np.float32)).astype(np.float32)
                seqs_block = create_sequences(scaled_block, SEQ_LENGTH)
                if len(seqs_block) == 0:
                    continue
                z_parts = []
                for s in range(0, len(seqs_block), BATCH_SIZE):
                    e = min(s + BATCH_SIZE, len(seqs_block))
                    x_t = torch.tensor(seqs_block[s:e], dtype=torch.float32).to(DEVICE)
                    z_parts.append(transformer.get_representation(x_t).cpu().numpy())
                z_block = np.concatenate(z_parts)
                scores_block = ocsvm.dissimilarity_score(z_block)
                train_scores_all.append(scores_block)
            except Exception as e:
                print(f"  WARNING: {os.path.basename(tf)}: {e}")

    if train_scores_all:
        all_train = np.concatenate(train_scores_all)
        tau = OCSVM.fit_baseline_tau(all_train, CONTAMINATION)
    else:
        all_train = np.array([0.0])
        tau = 0.0
    print(f"  τ = {tau:.6f}, n_train_scores = {len(all_train)}")

    # Test scores across all days
    all_test_scores = []
    for fpath in TEST_FILES:
        _, _, seqs_day = prepare_sequences(fpath, feat_names, scaler)
        z_parts = []
        with torch.no_grad():
            for s in range(0, len(seqs_day), BATCH_SIZE):
                e = min(s + BATCH_SIZE, len(seqs_day))
                x_t = torch.tensor(seqs_day[s:e], dtype=torch.float32).to(DEVICE)
                z_parts.append(transformer.get_representation(x_t).cpu().numpy())
        z_day = np.concatenate(z_parts)
        all_test_scores.append(ocsvm.dissimilarity_score(z_day))
    all_test_flat = np.concatenate(all_test_scores)

    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
    axes[0].hist(all_train, bins=200, color=BLUE, edgecolor="white", alpha=0.85, density=True)
    axes[0].axvline(0, color="gray", linestyle=":", linewidth=1, label="OC-SVM boundary ($f=0$)")
    axes[0].axvline(tau, color=RED, linestyle="--", linewidth=2,
                    label=f"$\\tau$ = {tau:.4f} (P{100*(1-CONTAMINATION):.0f})")
    axes[0].set_xlabel("Dissimilarity score (training)")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Training scores ({year}, {len(all_train):,} samples)")
    axes[0].legend(fontsize=7)

    axes[1].hist(all_test_flat, bins=200, color=ORANGE, edgecolor="white", alpha=0.85, density=True)
    axes[1].axvline(0, color="gray", linestyle=":", linewidth=1, label="OC-SVM boundary")
    axes[1].axvline(tau, color=RED, linestyle="--", linewidth=2, label=f"$\\tau$ = {tau:.4f}")
    axes[1].set_xlabel("Dissimilarity score (test)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Test scores with $\\tau$ overlay")
    axes[1].legend(fontsize=7)

    taus_sweep = np.linspace(
        min(all_train.min(), all_test_flat.min()),
        np.percentile(all_test_flat, 99.5), 500)
    rates_train = [100.0 * np.mean(all_train >= t) for t in taus_sweep]
    rates_test = [100.0 * np.mean(all_test_flat >= t) for t in taus_sweep]
    axes[2].plot(taus_sweep, rates_train, label="Training", color=BLUE, linewidth=1.5)
    axes[2].plot(taus_sweep, rates_test, label="Test", color=ORANGE, linewidth=1.5)
    axes[2].axvline(tau, color=RED, linestyle="--", linewidth=1.5, label=f"$\\tau$ = {tau:.4f}")
    axes[2].axvline(0, color="gray", linestyle=":", linewidth=1, label="$f=0$ boundary")
    axes[2].set_xlabel(r"Threshold $\tau$")
    axes[2].set_ylabel("Detection rate (%)")
    axes[2].set_title("Detection rate vs. threshold")
    axes[2].legend(fontsize=7)
    axes[2].set_yscale("log")
    axes[2].set_ylim(bottom=0.001)
    plt.tight_layout()
    save_fig(fig, f"tfocsvm_tau_analysis_{year}.pdf")

    # Return key numbers for LaTeX
    return {
        "gamma": ocsvm._gamma,
        "rho": float(ocsvm._rho.item()),
        "w_norm": float(ocsvm._w.norm().item()),
        "n_landmarks": int(ocsvm._landmarks.shape[0]),
        "latent_dim": int(z_all.shape[1]),
        "n_test_sequences": int(z_all.shape[0]),
        "latent_mean": float(z_np.mean()),
        "latent_std": float(z_np.std()),
        "dead_dims": int((dim_var < 1e-6).sum()),
        "median_kernel": float(np.median(kernel_vals)),
        "typical_kernel": float(typical_kernel),
        "effective_rank_95": effective_rank_95,
        "condition_number": float(eigvals_pos[0] / eigvals_pos[-1]) if len(eigvals_pos) > 1 else float("inf"),
        "n_outliers_test": n_outliers,
        "outlier_rate": float(100.0 * n_outliers / len(decision_vals)),
        "rec_dissim_corr": corr,
        "tau": float(tau),
        "n_train_scores": len(all_train),
        "pca_var_1": float(pca.explained_variance_ratio_[0]),
        "pca_var_2": float(pca.explained_variance_ratio_[1]),
    }


# ==============================================================================
# PNN Diagnostics
# ==============================================================================
def generate_pnn_diagnostics(year):
    print(f"\n{'='*60}")
    print(f"PNN Diagnostics — {year}")
    print(f"{'='*60}")
    results_dir = os.path.join("results", str(year))
    feat_names, scaler = load_features_and_scaler(results_dir, "pnn")
    num_features = len(feat_names)

    model = PNN(input_dim=num_features, hidden_dim=64).to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(results_dir, "pnn_weights.pth"),
        map_location=DEVICE, weights_only=True))
    model.eval()

    from detection.spoofing.gain import compute_spoofing_gains_batch

    SPOOF_Q, SPOOF_q = 4500, 100
    SPOOF_DELTA_A, SPOOF_DELTA_B = 0.0, 0.01
    SPOOF_FEES = {"maker": 0.0, "taker": 0.0008}

    # Run inference on first test file
    df_test, features_test, seqs = prepare_sequences(TEST_FILES[0], feat_names, scaler)

    all_mu, all_sigma, all_alpha = [], [], []
    with torch.no_grad():
        for start in range(0, len(seqs), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(seqs))
            x = torch.tensor(seqs[start:end, -1, :], dtype=torch.float32).to(DEVICE)
            mu, sigma, alpha = model(x)
            all_mu.append(mu.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())
            all_alpha.append(alpha.cpu().numpy())
    mu_arr = np.concatenate(all_mu).flatten()
    sigma_arr = np.concatenate(all_sigma).flatten()
    alpha_arr = np.concatenate(all_alpha).flatten()
    print(f"  Predictions: {len(mu_arr)} events")

    # 1. Predicted parameter distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    axes[0].hist(mu_arr, bins=150, color=BLUE, edgecolor="white", alpha=0.85, density=True)
    axes[0].set_xlabel(r"$\mu$ (location)")
    axes[0].set_ylabel("Density")
    axes[0].set_title(r"Predicted $\mu$")
    axes[0].axvline(mu_arr.mean(), color=RED, linestyle="--", label=f"mean={mu_arr.mean():.4f}")
    axes[0].legend()

    axes[1].hist(sigma_arr, bins=150, color=ORANGE, edgecolor="white", alpha=0.85, density=True)
    axes[1].set_xlabel(r"$\sigma$ (scale)")
    axes[1].set_ylabel("Density")
    axes[1].set_title(r"Predicted $\sigma$")
    axes[1].axvline(sigma_arr.mean(), color=RED, linestyle="--", label=f"mean={sigma_arr.mean():.4f}")
    axes[1].legend()

    axes[2].hist(alpha_arr, bins=150, color=GREEN, edgecolor="white", alpha=0.85, density=True)
    axes[2].set_xlabel(r"$\alpha$ (skewness)")
    axes[2].set_ylabel("Density")
    axes[2].set_title(r"Predicted $\alpha$")
    axes[2].axvline(alpha_arr.mean(), color=RED, linestyle="--", label=f"mean={alpha_arr.mean():.4f}")
    axes[2].legend()
    plt.tight_layout()
    save_fig(fig, f"pnn_parameters_{year}.pdf")

    # 2. NLL score distribution
    target_idx = feat_names.index("log_return")
    y_raw = features_test[feat_names].values[
        SEQ_LENGTH:SEQ_LENGTH + len(mu_arr), target_idx].astype(np.float32)

    with torch.no_grad():
        y_t = torch.tensor(y_raw).unsqueeze(1)
        mu_t = torch.tensor(mu_arr).unsqueeze(1)
        sigma_t = torch.tensor(sigma_arr).unsqueeze(1)
        alpha_t = torch.tensor(alpha_arr).unsqueeze(1)
        zz = (y_t - mu_t) / sigma_t
        phi_z = (1.0 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * zz**2)
        Phi_az = 0.5 * (1 + torch.erf(alpha_t * zz / math.sqrt(2)))
        pdf = (2.0 / sigma_t) * phi_z * Phi_az
        nll = -torch.log(pdf + 1e-10).squeeze().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    p1, p99 = np.percentile(nll, [1, 99.5])
    axes[0].hist(nll, bins=200, color=BLUE, edgecolor="white", alpha=0.85, density=True,
                 range=(p1, p99))
    axes[0].set_xlabel("NLL")
    axes[0].set_ylabel("Density")
    axes[0].set_title("NLL distribution (clipped to [P1, P99.5])")
    axes[0].axvline(np.percentile(nll, 99), color=RED, linestyle="--",
                    label=f"P99 = {np.percentile(nll, 99):.2f}")
    axes[0].legend()

    axes[1].scatter(sigma_arr, nll, s=1, alpha=0.1, color=BLUE)
    axes[1].set_xlabel(r"$\sigma$ (predicted scale)")
    axes[1].set_ylabel("NLL")
    axes[1].set_title(r"NLL vs. predicted $\sigma$")
    plt.tight_layout()
    save_fig(fig, f"pnn_nll_distribution_{year}.pdf")

    # 3. Spoofing gain distribution (subsampled)
    spread_raw = (df_test["ask-price-1"] - df_test["bid-price-1"]).values
    mid_price = 0.5 * (df_test["ask-price-1"] + df_test["bid-price-1"]).values
    spread_seq = np.abs(spread_raw[SEQ_LENGTH:SEQ_LENGTH + len(mu_arr)])
    mid_seq = mid_price[SEQ_LENGTH:SEQ_LENGTH + len(mu_arr)]
    spread_seq = np.where(spread_seq > 0, spread_seq, 1e-4)
    mu_eur = mu_arr * mid_seq
    sigma_eur = sigma_arr * mid_seq

    # Subsample for speed — 20K is plenty for histogram & statistics
    N_SUB = min(20_000, len(mu_eur))
    rng = np.random.default_rng(42)
    idx_sub = rng.choice(len(mu_eur), N_SUB, replace=False)
    mu_sub  = mu_eur[idx_sub]
    sig_sub = sigma_eur[idx_sub]
    alp_sub = alpha_arr[idx_sub]
    spr_sub = spread_seq[idx_sub]

    gains = compute_spoofing_gains_batch(
        mu_sub, sig_sub, alp_sub, spr_sub,
        delta_a=SPOOF_DELTA_A, delta_b=SPOOF_DELTA_B,
        Q=SPOOF_Q, q=SPOOF_q, fees=SPOOF_FEES, side="ask",
    )
    gains_clean = gains[np.isfinite(gains)]
    n_pos = int(np.sum(gains_clean > 0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    p1, p99 = np.nanpercentile(gains_clean, [0.5, 99.5])
    axes[0].hist(gains_clean, bins=200, color=BLUE, edgecolor="white", alpha=0.85,
                 density=True, range=(p1, p99))
    axes[0].axvline(0, color=RED, linestyle="--", linewidth=1.5, label="$G_t = 0$ (decision boundary)")
    axes[0].set_xlabel("Spoofing Gain (EUR)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("$G_t$ distribution (clipped to [P0.5, P99.5])")
    axes[0].legend(fontsize=8)

    axes[1].plot(gains, linewidth=0.3, alpha=0.7, color=BLUE)
    axes[1].axhline(0, color=RED, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Sequence index")
    axes[1].set_ylabel("Spoofing Gain (EUR)")
    axes[1].set_title("$G_t$ time series")
    plt.tight_layout()
    save_fig(fig, f"pnn_gain_distribution_{year}.pdf")

    # 4. δ_b sensitivity sweep
    delta_b_values = np.linspace(0.0, 0.05, 15)
    mean_gains = []
    pct_positive = []
    for db in delta_b_values:
        g = compute_spoofing_gains_batch(
            mu_sub, sig_sub, alp_sub, spr_sub,
            delta_a=SPOOF_DELTA_A, delta_b=db,
            Q=SPOOF_Q, q=SPOOF_q, fees=SPOOF_FEES, side="ask",
        )
        g_clean = g[np.isfinite(g)]
        mean_gains.append(np.nanmean(g_clean) if len(g_clean) > 0 else 0.0)
        pct_positive.append(100 * np.mean(g_clean > 0) if len(g_clean) > 0 else 0.0)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(delta_b_values, mean_gains, "o-", color=BLUE, markersize=3, label="Mean $G_t$")
    ax1.set_xlabel(r"$\delta_b$ (spoofing order distance)")
    ax1.set_ylabel("Mean $G_t$ (EUR)", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax1.axvline(SPOOF_DELTA_B, color=RED, linestyle="--", alpha=0.6,
                label=f"Baseline $\\delta_b$ = {SPOOF_DELTA_B}")
    ax2 = ax1.twinx()
    ax2.plot(delta_b_values, pct_positive, "s-", color=ORANGE, markersize=3,
             label="% flagged ($G_t > 0$)")
    ax2.set_ylabel("% flagged anomalous", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    plt.title(r"$\delta_b$ sensitivity sweep")
    plt.tight_layout()
    save_fig(fig, f"pnn_delta_b_sweep_{year}.pdf")

    return {
        "mu_mean": float(mu_arr.mean()),
        "mu_std": float(mu_arr.std()),
        "sigma_mean": float(sigma_arr.mean()),
        "sigma_std": float(sigma_arr.std()),
        "sigma_min": float(sigma_arr.min()),
        "alpha_mean": float(alpha_arr.mean()),
        "alpha_std": float(alpha_arr.std()),
        "nll_mean": float(nll.mean()),
        "nll_p99": float(np.percentile(nll, 99)),
        "n_predictions": len(mu_arr),
        "n_gains_positive": n_pos,
        "gains_pct_positive": float(100 * n_pos / len(gains_clean)),
        "gains_mean": float(gains_clean.mean()),
    }


# ==============================================================================
# PRAE Diagnostics
# ==============================================================================
def generate_prae_diagnostics(year):
    print(f"\n{'='*60}")
    print(f"PRAE Diagnostics — {year}")
    print(f"{'='*60}")
    results_dir = os.path.join("results", str(year))
    feat_names, scaler = load_features_and_scaler(results_dir, "prae")
    num_features = len(feat_names)

    # Load model
    weights_path = os.path.join(results_dir, "prae_weights.pth")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    num_train_samples = state_dict["mu"].shape[0]
    print(f"  Training samples (mu): {num_train_samples:,}")

    backbone = BottleneckTransformer(
        num_features=num_features, model_dim=128, num_heads=8,
        num_layers=6, representation_dim=128, sequence_length=SEQ_LENGTH,
        dim_feedforward=512,
    )
    model = PRAE(
        backbone_model=backbone,
        num_train_samples=num_train_samples,
        lambda_reg=1.0,
        sigma=0.5,
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    mu_values = model.mu.detach().cpu().numpy()

    # 1. μ distribution
    anomaly_threshold = 0.1
    n_anomalies = int((mu_values < anomaly_threshold).sum())
    pct_anomalies = 100.0 * n_anomalies / len(mu_values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(mu_values, bins=100, color=BLUE, edgecolor="white", alpha=0.85)
    axes[0].axvline(anomaly_threshold, color=RED, linestyle="--", linewidth=1.5,
                    label=f"Anomaly threshold ($\\mu$ < {anomaly_threshold})")
    axes[0].set_xlabel(r"$\mu$ value")
    axes[0].set_ylabel("Number of training samples")
    axes[0].set_title(r"Distribution of learned $\mu$ across training set")
    axes[0].legend(fontsize=8)

    low_mu = mu_values[mu_values < 0.3]
    if len(low_mu) > 0:
        axes[1].hist(low_mu, bins=60, color=ORANGE, edgecolor="white", alpha=0.85)
        axes[1].axvline(anomaly_threshold, color=RED, linestyle="--", linewidth=1.5,
                        label=f"$\\mu$ < {anomaly_threshold}")
        axes[1].set_xlabel(r"$\mu$ value")
        axes[1].set_ylabel("Number of training samples")
        axes[1].set_title(r"Zoom: $\mu$ < 0.3 (anomalous tail)")
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "No samples with $\\mu < 0.3$",
                     ha="center", va="center", transform=axes[1].transAxes)
    plt.tight_layout()
    save_fig(fig, f"prae_mu_distribution_{year}.pdf")

    # 2. Reconstruction error: gated-in vs gated-out
    from torch.utils.data import DataLoader, TensorDataset
    year_files = sorted(glob.glob(os.path.join(DATA_DIR, f"{year}*.parquet")))
    train_file_list = year_files[:len(year_files) - NUM_HOLDOUT]
    N_SUBSET = min(1, len(train_file_list))

    rec_errors_in, rec_errors_out = [], []
    mu_idx = 0
    for tf in train_file_list[:N_SUBSET]:
        print(f"  Loading training file: {os.path.basename(tf)}")
        _, feat_d = load_processed(tf, "xltime", LOB_COLUMNS)
        for c in feat_names:
            if c not in feat_d.columns:
                feat_d[c] = 0.0
        feat_d = feat_d[feat_names]
        sc = scaler.transform(feat_d.values.astype(np.float32)).astype(np.float32)
        seqs = create_sequences(sc, SEQ_LENGTH)
        if len(seqs) == 0:
            continue
        # Subsample sequences for speed (5000 max)
        if len(seqs) > 5000:
            sub_idx = np.random.choice(len(seqs), 5000, replace=False)
            sub_idx.sort()
            seqs = seqs[sub_idx]
            mu_offset_sub = sub_idx
        else:
            mu_offset_sub = np.arange(len(seqs))

        print(f"  Processing {len(seqs)} sequences through PRAE...")
        x_t = torch.tensor(seqs, dtype=torch.float32)
        ds = TensorDataset(x_t, x_t)
        loader = DataLoader(ds, batch_size=2048, shuffle=False)

        day_errs = []
        model.eval()
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                x = batch[0].to(DEVICE)
                rec, _ = model(x, training=False)
                err = torch.sum((x - rec) ** 2, dim=tuple(range(1, x.dim()))).cpu().numpy()
                day_errs.append(err)
                if bi % 5 == 0:
                    print(f"    batch {bi+1}/{len(loader)}")
        day_errs = np.concatenate(day_errs)

        # Map subsampled indices to mu values
        end_idx = mu_idx + len(mu_offset_sub)
        if mu_idx + mu_offset_sub[-1] < len(mu_values):
            day_mu = mu_values[mu_idx + mu_offset_sub]
            rec_errors_in.append(day_errs[day_mu >= anomaly_threshold])
            rec_errors_out.append(day_errs[day_mu < anomaly_threshold])
        mu_idx += len(feat_d) - SEQ_LENGTH + 1  # original seq count

    err_in = np.concatenate(rec_errors_in) if rec_errors_in else np.array([])
    err_out = np.concatenate(rec_errors_out) if rec_errors_out else np.array([])

    if len(err_in) > 0:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        clip_hi = np.percentile(np.concatenate([err_in, err_out]) if len(err_out) > 0 else err_in, 99)
        ax.hist(err_in, bins=150, alpha=0.7, color=BLUE,
                label=f"Gated-in ($\\mu \\geq$ {anomaly_threshold}, n={len(err_in):,})",
                density=True, range=(0, clip_hi))
        if len(err_out) > 0:
            ax.hist(err_out, bins=150, alpha=0.7, color=ORANGE,
                    label=f"Gated-out ($\\mu$ < {anomaly_threshold}, n={len(err_out):,})",
                    density=True, range=(0, clip_hi))
        ax.set_xlabel(r"Reconstruction Error $\sum(x - \hat{x})^2$")
        ax.set_ylabel("Density")
        ax.set_title("Reconstruction Error: Gated-In vs. Gated-Out (Training)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        save_fig(fig, f"prae_rec_error_gated_{year}.pdf")
    else:
        print("  WARNING: Could not compute gated reconstruction errors")

    # 3. Test-time score distribution
    print(f"  Computing test scores...")
    _, _, seqs_test = prepare_sequences(TEST_FILES[0], feat_names, scaler)
    # Subsample test sequences
    if len(seqs_test) > 5000:
        sub_i = np.random.choice(len(seqs_test), 5000, replace=False)
        seqs_test = seqs_test[sub_i]
    x_t = torch.tensor(seqs_test, dtype=torch.float32)
    ds = TensorDataset(x_t, x_t)
    loader = DataLoader(ds, batch_size=2048, shuffle=False)

    test_scores = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(DEVICE)
            rec, _ = model(x, training=False)
            err = torch.sum((x - rec) ** 2, dim=tuple(range(1, x.dim()))).cpu().numpy()
            test_scores.append(err)
    test_scores = np.concatenate(test_scores)

    fig, ax = plt.subplots(figsize=(8, 4))
    p99 = np.percentile(test_scores, 99.5)
    ax.hist(test_scores, bins=200, color=BLUE, edgecolor="white", alpha=0.85,
            density=True, range=(0, p99))
    ax.set_xlabel(r"Reconstruction Error $\sum(x-\hat{x})^2$")
    ax.set_ylabel("Density")
    ax.set_title(f"PRAE score distribution ({year})")
    ax.axvline(np.percentile(test_scores, 99), color=RED, linestyle="--",
               label=f"P99 = {np.percentile(test_scores, 99):.2f}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    save_fig(fig, f"prae_score_distribution_{year}.pdf")

    return {
        "n_train_samples": int(num_train_samples),
        "mu_mean": float(mu_values.mean()),
        "mu_std": float(mu_values.std()),
        "mu_min": float(mu_values.min()),
        "mu_max": float(mu_values.max()),
        "mu_median": float(np.median(mu_values)),
        "n_anomalies": n_anomalies,
        "pct_anomalies": pct_anomalies,
        "err_in_mean": float(err_in.mean()) if len(err_in) > 0 else None,
        "err_out_mean": float(err_out.mean()) if len(err_out) > 0 else None,
        "err_ratio": float(err_out.mean() / err_in.mean()) if len(err_out) > 0 and len(err_in) > 0 else None,
        "test_score_mean": float(test_scores.mean()),
        "test_score_p99": float(np.percentile(test_scores, 99)),
    }


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    import json

    np.random.seed(42)
    torch.manual_seed(42)

    # Check which figures already exist to skip completed work
    def _last_fig_exists(model, year):
        """Return True if the last figure for this model/year exists."""
        last_figs = {
            "tfocsvm": f"tfocsvm_tau_analysis_{year}.pdf",
            "pnn": f"pnn_delta_b_sweep_{year}.pdf",
            "prae": f"prae_score_distribution_{year}.pdf",
        }
        return os.path.exists(os.path.join(OUT_DIR, last_figs[model]))

    all_stats = {}
    for year in ["2015", "2017"]:
        stats = {}
        for model, func in [("tfocsvm", generate_tfocsvm_diagnostics),
                             ("pnn", generate_pnn_diagnostics),
                             ("prae", generate_prae_diagnostics)]:
            if _last_fig_exists(model, year):
                print(f"\n  [SKIP] {model.upper()} {year} — last figure already exists")
                stats[model] = {}
            else:
                stats[model] = func(year)
        all_stats[year] = stats

    # Save stats for LaTeX reference
    stats_path = os.path.join(OUT_DIR, "diagnostics_stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nAll statistics saved to {stats_path}")
    print("Done.")
