"""Compute cross-year comparison statistics for Section 5.1 paragraphs."""
import os, sys, glob, json, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detection.data.loaders import create_sequences, load_processed
from detection.data.preprocessing import split_first_hour_blocks
from detection.models.transformer import BottleneckTransformer
from detection.models.ocsvm import OCSVM
from detection.models.pnn import PNN
from detection.models.prae import PRAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join("data", "processed", "TOTF.PA-book")
SEQ_LENGTH = 25
BATCH_SIZE = 512
LOB_COLUMNS = [
    f"{side}-{typ}-{lvl}"
    for lvl in range(1, 11)
    for side, typ in [("bid","price"),("bid","volume"),("ask","price"),("ask","volume")]
]
FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
TEST_FILES = [FILES[22], FILES[23], FILES[24], FILES[25], FILES[26]]
NUM_HOLDOUT = 12
NUM_TAU_DAYS = 3
CONTAMINATION = 0.01

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

stats = {}

for year in [2015, 2017]:
    print(f"\n{'='*60}")
    print(f"Year: {year}")
    print(f"{'='*60}")
    results_dir = os.path.join("results", str(year))
    yr = {}

    print("  TF-OC-SVM...")
    feat_names, scaler = load_features_and_scaler(results_dir, "transformer_ocsvm")
    num_features = len(feat_names)

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

    dim_var = z_np.var(axis=0)
    yr["latent_mean_of_means"] = float(z_np.mean(axis=0).mean())
    yr["latent_mean_of_vars"] = float(dim_var.mean())
    yr["latent_median_var"] = float(np.median(dim_var))
    yr["latent_min_var"] = float(dim_var.min())
    yr["latent_max_var"] = float(dim_var.max())
    yr["latent_global_std"] = float(z_np.std())
    yr["dead_dims"] = int((dim_var < 1e-6).sum())

    decision_vals = ocsvm.decision_function(z_np)
    dissimilarity = -decision_vals
    yr["n_outliers_f0"] = int((decision_vals < 0).sum())
    yr["outlier_rate_f0"] = float(100.0 * yr["n_outliers_f0"] / len(decision_vals))

    rec_errors = []
    with torch.no_grad():
        for start in range(0, len(seqs), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(seqs))
            x_t = torch.tensor(seqs[start:end], dtype=torch.float32).to(DEVICE)
            x_hat = transformer(x_t)
            mse = ((x_hat - x_t) ** 2).mean(dim=(1, 2)).cpu().numpy()
            rec_errors.append(mse)
    rec_errors = np.concatenate(rec_errors)
    yr["rec_error_mean"] = float(rec_errors.mean())
    yr["rec_error_median"] = float(np.median(rec_errors))
    yr["rec_error_p99"] = float(np.percentile(rec_errors, 99))
    yr["rec_error_std"] = float(rec_errors.std())
    yr["rec_dissim_corr"] = float(np.corrcoef(rec_errors, dissimilarity)[0, 1])

    yr["dissim_mean"] = float(dissimilarity.mean())
    yr["dissim_std"] = float(dissimilarity.std())
    yr["dissim_median"] = float(np.median(dissimilarity))
    yr["dissim_p99"] = float(np.percentile(dissimilarity, 99))

    yr["gamma"] = float(ocsvm._gamma)
    yr["rho"] = float(ocsvm._rho.item())
    yr["w_norm"] = float(ocsvm._w.norm().item())

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_np)
    yr["pca_var_1"] = float(pca.explained_variance_ratio_[0])
    yr["pca_var_2"] = float(pca.explained_variance_ratio_[1])

    test_scores_path = os.path.join(results_dir, "test_output", "transformer_ocsvm_scores.npy")
    test_scores_all = np.load(test_scores_path)
    
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
                print(f"    WARNING: {os.path.basename(tf)}: {e}")
    
    if train_scores_all:
        all_train = np.concatenate(train_scores_all)
        tau = OCSVM.fit_baseline_tau(all_train, CONTAMINATION)
    else:
        all_train = np.array([0.0])
        tau = 0.0
    
    yr["tau"] = float(tau)
    yr["n_train_scores"] = len(all_train)
    yr["test_pct_above_tau"] = float(100.0 * np.mean(test_scores_all >= tau))
    yr["test_n_above_tau"] = int(np.sum(test_scores_all >= tau))
    yr["test_total"] = len(test_scores_all)

    print(f"    tau={tau:.6f}, test_pct_above_tau={yr['test_pct_above_tau']:.4f}%")
    print(f"    rec_error: mean={yr['rec_error_mean']:.4f}, p99={yr['rec_error_p99']:.4f}")
    print(f"    dissim: mean={yr['dissim_mean']:.4f}, p99={yr['dissim_p99']:.4f}")
    print(f"    rec_dissim_corr={yr['rec_dissim_corr']:.4f}")
    print(f"    latent var: mean={yr['latent_mean_of_vars']:.4f}, range=[{yr['latent_min_var']:.4f}, {yr['latent_max_var']:.4f}]")

    print("  PNN...")
    feat_names_pnn, scaler_pnn = load_features_and_scaler(results_dir, "pnn")
    model_pnn = PNN(input_dim=len(feat_names_pnn), hidden_dim=64).to(DEVICE)
    model_pnn.load_state_dict(torch.load(
        os.path.join(results_dir, "pnn_weights.pth"),
        map_location=DEVICE, weights_only=True))
    model_pnn.eval()

    df_test, features_test, seqs_pnn = prepare_sequences(TEST_FILES[0], feat_names_pnn, scaler_pnn)
    all_mu, all_sigma, all_alpha = [], [], []
    with torch.no_grad():
        for start in range(0, len(seqs_pnn), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(seqs_pnn))
            x = torch.tensor(seqs_pnn[start:end, -1, :], dtype=torch.float32).to(DEVICE)
            mu, sigma, alpha = model_pnn(x)
            all_mu.append(mu.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())
            all_alpha.append(alpha.cpu().numpy())
    mu_arr = np.concatenate(all_mu).flatten()
    sigma_arr = np.concatenate(all_sigma).flatten()
    alpha_arr = np.concatenate(all_alpha).flatten()

    target_idx = feat_names_pnn.index("log_return")
    y_raw = features_test[feat_names_pnn].values[
        SEQ_LENGTH:SEQ_LENGTH + len(mu_arr), target_idx].astype(np.float32)

    y_t = torch.tensor(y_raw).unsqueeze(1)
    mu_t = torch.tensor(mu_arr).unsqueeze(1)
    sigma_t = torch.tensor(sigma_arr).unsqueeze(1)
    alpha_t = torch.tensor(alpha_arr).unsqueeze(1)
    zz = (y_t - mu_t) / sigma_t
    phi_z = (1.0 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * zz**2)
    Phi_az = 0.5 * (1 + torch.erf(alpha_t * zz / math.sqrt(2)))
    pdf = (2.0 / sigma_t) * phi_z * Phi_az
    nll = -torch.log(pdf + 1e-10).squeeze().numpy()

    yr["pnn_mu_mean"] = float(mu_arr.mean())
    yr["pnn_mu_std"] = float(mu_arr.std())
    yr["pnn_sigma_mean"] = float(sigma_arr.mean())
    yr["pnn_sigma_std"] = float(sigma_arr.std())
    yr["pnn_sigma_min"] = float(sigma_arr.min())
    yr["pnn_alpha_mean"] = float(alpha_arr.mean())
    yr["pnn_alpha_std"] = float(alpha_arr.std())
    yr["pnn_nll_mean"] = float(nll.mean())
    yr["pnn_nll_median"] = float(np.median(nll))
    yr["pnn_nll_p99"] = float(np.percentile(nll, 99))
    yr["pnn_nll_p95"] = float(np.percentile(nll, 95))

    from scipy.special import erf as sp_erf, owens_t
    def _fast_skewnorm_cdf(x, mu, sigma, alpha):
        z = (np.asarray(x, dtype=np.float64) - mu) / sigma
        phi_z = 0.5 * (1.0 + sp_erf(z / np.sqrt(2.0)))
        return phi_z - 2.0 * owens_t(z, np.asarray(alpha, dtype=np.float64))
    import detection.spoofing.gain as _gain_mod
    _gain_mod.skewed_gaussian_cdf = _fast_skewnorm_cdf
    from detection.spoofing.gain import compute_spoofing_gains_batch

    SPOOF_Q, SPOOF_q = 4500, 100
    SPOOF_DELTA_A, SPOOF_DELTA_B = 0.0, 0.01
    SPOOF_FEES = {"maker": 0.0, "taker": 0.0008}

    spread_raw = (df_test["ask-price-1"] - df_test["bid-price-1"]).values
    mid_price = 0.5 * (df_test["ask-price-1"] + df_test["bid-price-1"]).values
    spread_seq = np.abs(spread_raw[SEQ_LENGTH:SEQ_LENGTH + len(mu_arr)])
    mid_seq = mid_price[SEQ_LENGTH:SEQ_LENGTH + len(mu_arr)]
    spread_seq = np.where(spread_seq > 0, spread_seq, 1e-4)
    mu_eur = mu_arr * mid_seq
    sigma_eur = sigma_arr * mid_seq

    N_SUB = min(20_000, len(mu_eur))
    rng = np.random.default_rng(42)
    idx_sub = rng.choice(len(mu_eur), N_SUB, replace=False)

    gains = compute_spoofing_gains_batch(
        mu_eur[idx_sub], sigma_eur[idx_sub], alpha_arr[idx_sub], spread_seq[idx_sub],
        delta_a=SPOOF_DELTA_A, delta_b=SPOOF_DELTA_B,
        Q=SPOOF_Q, q=SPOOF_q, fees=SPOOF_FEES, side="ask",
    )
    gains_clean = gains[np.isfinite(gains)]
    yr["pnn_gain_mean"] = float(gains_clean.mean())
    yr["pnn_gain_median"] = float(np.median(gains_clean))
    yr["pnn_gain_p99"] = float(np.percentile(gains_clean, 99))
    yr["pnn_gain_pct_positive"] = float(100 * np.mean(gains_clean > 0))
    yr["pnn_gain_n_positive"] = int(np.sum(gains_clean > 0))

    print(f"    sigma: mean={yr['pnn_sigma_mean']:.6f}, std={yr['pnn_sigma_std']:.6f}")
    print(f"    alpha: mean={yr['pnn_alpha_mean']:.4f}, std={yr['pnn_alpha_std']:.4f}")
    print(f"    nll: mean={yr['pnn_nll_mean']:.2f}, p99={yr['pnn_nll_p99']:.2f}")
    print(f"    gain: pct_positive={yr['pnn_gain_pct_positive']:.3f}%, mean={yr['pnn_gain_mean']:.2f}")

    print("  PRAE...")
    feat_names_prae, scaler_prae = load_features_and_scaler(results_dir, "prae")
    weights_path = os.path.join(results_dir, "prae_weights.pth")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    num_train_samples = state_dict["mu"].shape[0]
    mu_values = state_dict["mu"].numpy()

    yr["prae_n_train"] = int(num_train_samples)
    yr["prae_mu_mean"] = float(mu_values.mean())
    yr["prae_mu_std"] = float(mu_values.std())
    yr["prae_mu_min"] = float(mu_values.min())
    yr["prae_mu_max"] = float(mu_values.max())

    backbone = BottleneckTransformer(
        num_features=len(feat_names_prae), model_dim=128, num_heads=8,
        num_layers=6, representation_dim=128, sequence_length=SEQ_LENGTH,
        dim_feedforward=512,
    )
    model_prae = PRAE(
        backbone_model=backbone,
        num_train_samples=num_train_samples,
        lambda_reg=1.0,
        sigma=0.5,
    )
    model_prae.load_state_dict(state_dict)
    model_prae.to(DEVICE)
    model_prae.eval()

    _, _, seqs_prae = prepare_sequences(TEST_FILES[0], feat_names_prae, scaler_prae)
    if len(seqs_prae) > 10000:
        sub_i = np.random.choice(len(seqs_prae), 10000, replace=False)
        seqs_prae = seqs_prae[sub_i]
    
    test_scores_prae = []
    from torch.utils.data import DataLoader, TensorDataset
    x_t = torch.tensor(seqs_prae, dtype=torch.float32)
    ds = TensorDataset(x_t, x_t)
    loader = DataLoader(ds, batch_size=2048, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(DEVICE)
            rec, _ = model_prae(x, training=False)
            err = torch.sum((x - rec) ** 2, dim=tuple(range(1, x.dim()))).cpu().numpy()
            test_scores_prae.append(err)
    test_scores_prae = np.concatenate(test_scores_prae)

    yr["prae_score_mean"] = float(test_scores_prae.mean())
    yr["prae_score_median"] = float(np.median(test_scores_prae))
    yr["prae_score_p99"] = float(np.percentile(test_scores_prae, 99))
    yr["prae_score_std"] = float(test_scores_prae.std())

    print(f"    mu: mean={yr['prae_mu_mean']:.6f}, std={yr['prae_mu_std']:.6f}")
    print(f"    test score: mean={yr['prae_score_mean']:.2f}, p99={yr['prae_score_p99']:.2f}")

    stats[str(year)] = yr

print("\n" + "="*60)
print("CROSS-YEAR COMPARISON SUMMARY")
print("="*60)

s15, s17 = stats["2015"], stats["2017"]

print("\n--- TF-OC-SVM ---")
print(f"  Rec error mean: 2015={s15['rec_error_mean']:.4f}, 2017={s17['rec_error_mean']:.4f}")
print(f"  Rec error P99:  2015={s15['rec_error_p99']:.4f}, 2017={s17['rec_error_p99']:.4f}")
print(f"  Dissimilarity mean: 2015={s15['dissim_mean']:.4f}, 2017={s17['dissim_mean']:.4f}")
print(f"  Dissimilarity P99:  2015={s15['dissim_p99']:.4f}, 2017={s17['dissim_p99']:.4f}")
print(f"  Rec-dissim corr: 2015={s15['rec_dissim_corr']:.4f}, 2017={s17['rec_dissim_corr']:.4f}")
print(f"  Tau: 2015={s15['tau']:.6f}, 2017={s17['tau']:.6f}")
print(f"  Test % above tau: 2015={s15['test_pct_above_tau']:.4f}%, 2017={s17['test_pct_above_tau']:.4f}%")
print(f"  Latent var range: 2015=[{s15['latent_min_var']:.4f},{s15['latent_max_var']:.4f}], 2017=[{s17['latent_min_var']:.4f},{s17['latent_max_var']:.4f}]")
print(f"  PCA var explained: 2015=({s15['pca_var_1']:.3f},{s15['pca_var_2']:.3f}), 2017=({s17['pca_var_1']:.3f},{s17['pca_var_2']:.3f})")
print(f"  gamma: 2015={s15['gamma']:.6f}, 2017={s17['gamma']:.6f}")
print(f"  rho: 2015={s15['rho']:.4f}, 2017={s17['rho']:.4f}")
print(f"  w_norm: 2015={s15['w_norm']:.4f}, 2017={s17['w_norm']:.4f}")
print(f"  Outliers (f<0): 2015={s15['n_outliers_f0']}({s15['outlier_rate_f0']:.3f}%), 2017={s17['n_outliers_f0']}({s17['outlier_rate_f0']:.3f}%)")

print("\n--- PNN ---")
print(f"  Sigma mean: 2015={s15['pnn_sigma_mean']:.6f}, 2017={s17['pnn_sigma_mean']:.6f}")
print(f"  Alpha mean: 2015={s15['pnn_alpha_mean']:.4f}, 2017={s17['pnn_alpha_mean']:.4f}")
print(f"  Alpha std: 2015={s15['pnn_alpha_std']:.4f}, 2017={s17['pnn_alpha_std']:.4f}")
print(f"  NLL mean: 2015={s15['pnn_nll_mean']:.2f}, 2017={s17['pnn_nll_mean']:.2f}")
print(f"  NLL P99: 2015={s15['pnn_nll_p99']:.2f}, 2017={s17['pnn_nll_p99']:.2f}")
print(f"  Gain % positive: 2015={s15['pnn_gain_pct_positive']:.3f}%, 2017={s17['pnn_gain_pct_positive']:.3f}%")
print(f"  Gain mean: 2015={s15['pnn_gain_mean']:.2f}, 2017={s17['pnn_gain_mean']:.2f}")
print(f"  Gain P99: 2015={s15['pnn_gain_p99']:.2f}, 2017={s17['pnn_gain_p99']:.2f}")

print("\n--- PRAE ---")
print(f"  n_train: 2015={s15['prae_n_train']}, 2017={s17['prae_n_train']}")
print(f"  Mu mean: 2015={s15['prae_mu_mean']:.6f}, 2017={s17['prae_mu_mean']:.6f}")
print(f"  Mu std: 2015={s15['prae_mu_std']:.6f}, 2017={s17['prae_mu_std']:.6f}")
print(f"  Mu range: 2015=[{s15['prae_mu_min']:.6f},{s15['prae_mu_max']:.6f}], 2017=[{s17['prae_mu_min']:.6f},{s17['prae_mu_max']:.6f}]")
print(f"  Score mean: 2015={s15['prae_score_mean']:.2f}, 2017={s17['prae_score_mean']:.2f}")
print(f"  Score P99: 2015={s15['prae_score_p99']:.2f}, 2017={s17['prae_score_p99']:.2f}")
print(f"  Score std: 2015={s15['prae_score_std']:.2f}, 2017={s17['prae_score_std']:.2f}")

out = os.path.join("figures", "diagnostics", "year_comparison_stats.json")
with open(out, "w") as f:
    json.dump(stats, f, indent=2)
print(f"\nSaved: {out}")
