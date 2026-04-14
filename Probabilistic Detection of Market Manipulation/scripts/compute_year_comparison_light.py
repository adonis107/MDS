"""Lightweight cross-year comparison using pre-computed artifacts only (no inference)."""
import os, sys, json, glob
import numpy as np
import torch
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_DIR = os.path.join("data", "processed", "TOTF.PA-book")
CONTAMINATION = 0.01
stats = {}

for year in [2015, 2017]:
    print(f"\n{'='*60}\nYear: {year}\n{'='*60}")
    results_dir = os.path.join("results", str(year))
    test_dir = os.path.join(results_dir, "test_output")
    yr = {}

    print("  TF-OC-SVM...")
    ocsvm = torch.load(
        os.path.join(results_dir, "transformer_ocsvm_detector.pth"),
        map_location="cpu", weights_only=False)
    for attr in ['_landmarks', '_w', '_rho', '_normalization']:
        if hasattr(ocsvm, attr):
            val = getattr(ocsvm, attr)
            if isinstance(val, torch.Tensor):
                setattr(ocsvm, attr, val.cpu())
    yr["gamma"] = float(ocsvm._gamma)
    yr["rho"] = float(ocsvm._rho.item())
    yr["w_norm"] = float(ocsvm._w.norm().item())
    yr["n_landmarks"] = int(ocsvm._landmarks.shape[0])

    K_CC = ocsvm._rbf_kernel(ocsvm._landmarks, ocsvm._landmarks)
    eigvals = np.linalg.eigvalsh(K_CC.cpu().numpy())
    eigvals = np.sort(eigvals)[::-1]
    eigvals_pos = eigvals[eigvals > 0]
    cumvar = np.cumsum(eigvals_pos) / np.sum(eigvals_pos)
    yr["effective_rank_95"] = int(np.searchsorted(cumvar, 0.95) + 1)
    yr["condition_number"] = float(eigvals_pos[0] / eigvals_pos[-1]) if len(eigvals_pos) > 1 else float("inf")

    tfocsvm_scores = np.load(os.path.join(test_dir, "transformer_ocsvm_scores.npy"))
    yr["dissim_mean"] = float(tfocsvm_scores.mean())
    yr["dissim_std"] = float(tfocsvm_scores.std())
    yr["dissim_median"] = float(np.median(tfocsvm_scores))
    yr["dissim_p95"] = float(np.percentile(tfocsvm_scores, 95))
    yr["dissim_p99"] = float(np.percentile(tfocsvm_scores, 99))
    yr["n_test_total"] = int(len(tfocsvm_scores))

    yr["pct_above_zero"] = float(100 * np.mean(tfocsvm_scores > 0))

    from detection.models.ocsvm import OCSVM as _OCSVM
    from detection.data.loaders import create_sequences, load_processed
    from detection.data.preprocessing import split_first_hour_blocks
    LOB_COLUMNS = [
        f"{side}-{typ}-{lvl}"
        for lvl in range(1, 11)
        for side, typ in [("bid","price"),("bid","volume"),("ask","price"),("ask","volume")]
    ]
    SEQ_LENGTH = 25
    BATCH_SIZE = 512
    from detection.models.transformer import BottleneckTransformer

    feat_path = os.path.join(results_dir, "transformer_ocsvm_features.txt")
    with open(feat_path) as f:
        feat_names = [line.strip() for line in f if line.strip()]
    sc = joblib.load(os.path.join(results_dir, "transformer_ocsvm_scaler.pkl"))

    transformer = BottleneckTransformer(
        num_features=len(feat_names), model_dim=128, num_heads=8,
        num_layers=6, representation_dim=128, sequence_length=SEQ_LENGTH,
        dim_feedforward=512,
    )
    state = torch.load(
        os.path.join(results_dir, "transformer_ocsvm_weights.pth"),
        map_location="cpu", weights_only=True)
    state = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
    transformer.load_state_dict(state)
    transformer.eval().cpu()

    year_files = sorted(glob.glob(os.path.join(DATA_DIR, f"{year}*.parquet")))
    train_files = year_files[:len(year_files) - 12]
    train_scores_all = []
    for tf in train_files[:3]:
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
            scaled_block = sc.transform(
                train_block.values.astype(np.float32)).astype(np.float32)
            seqs_block = create_sequences(scaled_block, SEQ_LENGTH)
            if len(seqs_block) == 0:
                continue
            if len(seqs_block) > 5000:
                idx = np.random.choice(len(seqs_block), 5000, replace=False)
                seqs_block = seqs_block[idx]
            with torch.no_grad():
                z_parts = []
                for s in range(0, len(seqs_block), BATCH_SIZE):
                    e = min(s + BATCH_SIZE, len(seqs_block))
                    x_t = torch.tensor(seqs_block[s:e], dtype=torch.float32)
                    z_parts.append(transformer.get_representation(x_t).numpy())
                z_block = np.concatenate(z_parts)
                scores_block = ocsvm.dissimilarity_score(z_block)
                train_scores_all.append(scores_block)
                print(f"    {os.path.basename(tf)}: {len(scores_block)} scores")
        except Exception as e:
            print(f"    WARNING: {os.path.basename(tf)}: {e}")

    if train_scores_all:
        all_train = np.concatenate(train_scores_all)
        tau = _OCSVM.fit_baseline_tau(all_train, CONTAMINATION)
        yr["tau"] = float(tau)
        yr["n_train_scores"] = len(all_train)
    else:
        yr["tau"] = 0.0
        yr["n_train_scores"] = 0

    yr["test_pct_above_tau"] = float(100 * np.mean(tfocsvm_scores >= yr["tau"]))
    yr["test_n_above_tau"] = int(np.sum(tfocsvm_scores >= yr["tau"]))

    FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    TEST_FILES = [FILES[22], FILES[23], FILES[24], FILES[25], FILES[26]]

    df_t, feat_t = load_processed(TEST_FILES[0], "xltime", LOB_COLUMNS)
    for c in feat_names:
        if c not in feat_t.columns:
            feat_t[c] = 0.0
    feat_t = feat_t[feat_names]
    scaled_t = sc.transform(feat_t.values.astype(np.float32)).astype(np.float32)
    seqs_test = create_sequences(scaled_t, SEQ_LENGTH)
    if len(seqs_test) > 5000:
        sub_idx = np.random.choice(len(seqs_test), 5000, replace=False)
        seqs_test = seqs_test[sub_idx]
    print(f"    Test day: {len(seqs_test)} sequences (subsampled)")

    with torch.no_grad():
        all_z = []
        all_rec_err = []
        for s in range(0, len(seqs_test), BATCH_SIZE):
            e = min(s + BATCH_SIZE, len(seqs_test))
            x_t = torch.tensor(seqs_test[s:e], dtype=torch.float32)
            z = transformer.get_representation(x_t)
            x_hat = transformer(x_t)
            mse = ((x_hat - x_t) ** 2).mean(dim=(1, 2)).numpy()
            all_z.append(z.numpy())
            all_rec_err.append(mse)

    z_np = np.concatenate(all_z)
    rec_errors = np.concatenate(all_rec_err)

    dim_var = z_np.var(axis=0)
    yr["latent_mean_of_vars"] = float(dim_var.mean())
    yr["latent_median_var"] = float(np.median(dim_var))
    yr["latent_min_var"] = float(dim_var.min())
    yr["latent_max_var"] = float(dim_var.max())
    yr["latent_global_std"] = float(z_np.std())
    yr["dead_dims"] = int((dim_var < 1e-6).sum())

    yr["rec_error_mean"] = float(rec_errors.mean())
    yr["rec_error_median"] = float(np.median(rec_errors))
    yr["rec_error_p99"] = float(np.percentile(rec_errors, 99))
    yr["rec_error_std"] = float(rec_errors.std())

    decision_vals = ocsvm.decision_function(z_np)
    dissim_sub = -decision_vals
    yr["rec_dissim_corr"] = float(np.corrcoef(rec_errors, dissim_sub)[0, 1])

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(z_np)
    yr["pca_var_1"] = float(pca.explained_variance_ratio_[0])
    yr["pca_var_2"] = float(pca.explained_variance_ratio_[1])

    del transformer

    print(f"    tau={yr['tau']:.6f}, test_pct_above_tau={yr['test_pct_above_tau']:.4f}%")
    print(f"    rec_error: mean={yr['rec_error_mean']:.6f}, p99={yr['rec_error_p99']:.6f}")
    print(f"    rec_dissim_corr={yr['rec_dissim_corr']:.4f}")
    print(f"    latent var: mean={yr['latent_mean_of_vars']:.4f}")
    print(f"    PCA: ({yr['pca_var_1']:.4f}, {yr['pca_var_2']:.4f})")

    print("  PNN...")
    pnn_scores = np.load(os.path.join(test_dir, "pnn_scores.npy"))
    yr["pnn_score_mean"] = float(pnn_scores.mean())
    yr["pnn_score_std"] = float(pnn_scores.std())
    yr["pnn_score_median"] = float(np.median(pnn_scores))
    yr["pnn_score_p95"] = float(np.percentile(pnn_scores, 95))
    yr["pnn_score_p99"] = float(np.percentile(pnn_scores, 99))
    yr["pnn_n_total"] = int(len(pnn_scores))

    feat_path_pnn = os.path.join(results_dir, "pnn_features.txt")
    with open(feat_path_pnn) as f:
        feat_names_pnn = [line.strip() for line in f if line.strip()]
    sc_pnn = joblib.load(os.path.join(results_dir, "pnn_scaler.pkl"))

    from detection.models.pnn import PNN as _PNN
    model_pnn = _PNN(input_dim=len(feat_names_pnn), hidden_dim=64)
    pnn_state = torch.load(
        os.path.join(results_dir, "pnn_weights.pth"),
        map_location="cpu", weights_only=True)
    pnn_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in pnn_state.items()}
    model_pnn.load_state_dict(pnn_state)
    model_pnn.eval().cpu()

    df_pnn, feat_pnn = load_processed(TEST_FILES[0], "xltime", LOB_COLUMNS)
    for c in feat_names_pnn:
        if c not in feat_pnn.columns:
            feat_pnn[c] = 0.0
    feat_pnn = feat_pnn[feat_names_pnn]
    scaled_pnn = sc_pnn.transform(feat_pnn.values.astype(np.float32)).astype(np.float32)
    seqs_pnn = create_sequences(scaled_pnn, SEQ_LENGTH)

    N_PNN = min(20000, len(seqs_pnn))
    pnn_idx = np.random.choice(len(seqs_pnn), N_PNN, replace=False)
    seqs_pnn_sub = seqs_pnn[pnn_idx]

    with torch.no_grad():
        all_mu, all_sigma, all_alpha = [], [], []
        for s in range(0, len(seqs_pnn_sub), BATCH_SIZE):
            e = min(s + BATCH_SIZE, len(seqs_pnn_sub))
            x = torch.tensor(seqs_pnn_sub[s:e, -1, :], dtype=torch.float32)
            mu, sigma, alpha = model_pnn(x)
            all_mu.append(mu.numpy())
            all_sigma.append(sigma.numpy())
            all_alpha.append(alpha.numpy())
    mu_arr = np.concatenate(all_mu).flatten()
    sigma_arr = np.concatenate(all_sigma).flatten()
    alpha_arr = np.concatenate(all_alpha).flatten()

    yr["pnn_mu_mean"] = float(mu_arr.mean())
    yr["pnn_mu_std"] = float(mu_arr.std())
    yr["pnn_sigma_mean"] = float(sigma_arr.mean())
    yr["pnn_sigma_std"] = float(sigma_arr.std())
    yr["pnn_sigma_min"] = float(sigma_arr.min())
    yr["pnn_alpha_mean"] = float(alpha_arr.mean())
    yr["pnn_alpha_std"] = float(alpha_arr.std())

    import math
    target_idx = feat_names_pnn.index("log_return")
    y_raw = feat_pnn.values[SEQ_LENGTH + pnn_idx, target_idx].astype(np.float32)
    zz = (y_raw - mu_arr) / sigma_arr
    phi_z = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * zz**2)
    Phi_az = 0.5 * (1 + np.vectorize(math.erf)(alpha_arr * zz / np.sqrt(2)))
    pdf = (2.0 / sigma_arr) * phi_z * Phi_az
    nll = -np.log(pdf + 1e-10)
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

    spread_raw = (df_pnn["ask-price-1"] - df_pnn["bid-price-1"]).values
    mid_price = 0.5 * (df_pnn["ask-price-1"] + df_pnn["bid-price-1"]).values
    actual_idx = SEQ_LENGTH + pnn_idx
    spread_seq = np.abs(spread_raw[actual_idx])
    mid_seq = mid_price[actual_idx]
    spread_seq = np.where(spread_seq > 0, spread_seq, 1e-4)
    mu_eur = mu_arr * mid_seq
    sigma_eur = sigma_arr * mid_seq

    gains = compute_spoofing_gains_batch(
        mu_eur, sigma_eur, alpha_arr, spread_seq,
        delta_a=0.0, delta_b=0.01,
        Q=4500, q=100, fees={"maker": 0.0, "taker": 0.0008}, side="ask",
    )
    gains_clean = gains[np.isfinite(gains)]
    yr["pnn_gain_mean"] = float(gains_clean.mean())
    yr["pnn_gain_median"] = float(np.median(gains_clean))
    yr["pnn_gain_p99"] = float(np.percentile(gains_clean, 99))
    yr["pnn_gain_pct_positive"] = float(100 * np.mean(gains_clean > 0))
    yr["pnn_gain_n_positive"] = int(np.sum(gains_clean > 0))

    del model_pnn
    print(f"    sigma: mean={yr['pnn_sigma_mean']:.6f}, std={yr['pnn_sigma_std']:.6f}")
    print(f"    alpha: mean={yr['pnn_alpha_mean']:.4f}, std={yr['pnn_alpha_std']:.4f}")
    print(f"    nll: mean={yr['pnn_nll_mean']:.2f}, p99={yr['pnn_nll_p99']:.2f}")
    print(f"    gain: pct_positive={yr['pnn_gain_pct_positive']:.3f}%, mean={yr['pnn_gain_mean']:.2f}")

    print("  PRAE...")
    prae_scores = np.load(os.path.join(test_dir, "prae_scores.npy"))
    yr["prae_test_score_mean"] = float(prae_scores.mean())
    yr["prae_test_score_std"] = float(prae_scores.std())
    yr["prae_test_score_median"] = float(np.median(prae_scores))
    yr["prae_test_score_p95"] = float(np.percentile(prae_scores, 95))
    yr["prae_test_score_p99"] = float(np.percentile(prae_scores, 99))
    yr["prae_n_test"] = int(len(prae_scores))

    state_dict = torch.load(
        os.path.join(results_dir, "prae_weights.pth"),
        map_location="cpu", weights_only=True)
    mu_values = state_dict["mu"].numpy()
    yr["prae_n_train"] = int(len(mu_values))
    yr["prae_mu_mean"] = float(mu_values.mean())
    yr["prae_mu_std"] = float(mu_values.std())
    yr["prae_mu_min"] = float(mu_values.min())
    yr["prae_mu_max"] = float(mu_values.max())
    yr["prae_mu_range"] = float(mu_values.max() - mu_values.min())

    print(f"    n_train={yr['prae_n_train']}, mu: mean={yr['prae_mu_mean']:.6f}, std={yr['prae_mu_std']:.6f}")
    print(f"    test score: mean={yr['prae_test_score_mean']:.2f}, p99={yr['prae_test_score_p99']:.2f}")

    stats[str(year)] = yr

print("\n" + "="*60)
print("CROSS-YEAR COMPARISON SUMMARY")
print("="*60)

s15, s17 = stats["2015"], stats["2017"]

print("\n--- TF-OC-SVM ---")
print(f"  gamma: 2015={s15['gamma']:.6f}, 2017={s17['gamma']:.6f}")
print(f"  rho: 2015={s15['rho']:.4f}, 2017={s17['rho']:.4f}")
print(f"  w_norm: 2015={s15['w_norm']:.4f}, 2017={s17['w_norm']:.4f}")
print(f"  ER95: 2015={s15['effective_rank_95']}, 2017={s17['effective_rank_95']}")
print(f"  Rec error mean: 2015={s15['rec_error_mean']:.6f}, 2017={s17['rec_error_mean']:.6f}")
print(f"  Rec error P99:  2015={s15['rec_error_p99']:.6f}, 2017={s17['rec_error_p99']:.6f}")
print(f"  Dissim mean: 2015={s15['dissim_mean']:.6f}, 2017={s17['dissim_mean']:.6f}")
print(f"  Dissim P99:  2015={s15['dissim_p99']:.6f}, 2017={s17['dissim_p99']:.6f}")
print(f"  Dissim % > 0: 2015={s15['pct_above_zero']:.3f}%, 2017={s17['pct_above_zero']:.3f}%")
print(f"  Rec-dissim corr: 2015={s15['rec_dissim_corr']:.4f}, 2017={s17['rec_dissim_corr']:.4f}")
print(f"  Tau: 2015={s15['tau']:.6f}, 2017={s17['tau']:.6f}")
print(f"  Test % >= tau: 2015={s15['test_pct_above_tau']:.4f}%, 2017={s17['test_pct_above_tau']:.4f}%")
print(f"  Latent var mean: 2015={s15['latent_mean_of_vars']:.4f}, 2017={s17['latent_mean_of_vars']:.4f}")
print(f"  PCA: 2015=({s15['pca_var_1']:.4f},{s15['pca_var_2']:.4f}), 2017=({s17['pca_var_1']:.4f},{s17['pca_var_2']:.4f})")

print("\n--- PNN ---")
print(f"  mu mean: 2015={s15['pnn_mu_mean']:.6f}, 2017={s17['pnn_mu_mean']:.6f}")
print(f"  sigma mean: 2015={s15['pnn_sigma_mean']:.6f}, 2017={s17['pnn_sigma_mean']:.6f}")
print(f"  sigma std: 2015={s15['pnn_sigma_std']:.6f}, 2017={s17['pnn_sigma_std']:.6f}")
print(f"  alpha mean: 2015={s15['pnn_alpha_mean']:.4f}, 2017={s17['pnn_alpha_mean']:.4f}")
print(f"  alpha std: 2015={s15['pnn_alpha_std']:.4f}, 2017={s17['pnn_alpha_std']:.4f}")
print(f"  NLL mean: 2015={s15['pnn_nll_mean']:.2f}, 2017={s17['pnn_nll_mean']:.2f}")
print(f"  NLL P99: 2015={s15['pnn_nll_p99']:.2f}, 2017={s17['pnn_nll_p99']:.2f}")
print(f"  Gain % positive: 2015={s15['pnn_gain_pct_positive']:.3f}%, 2017={s17['pnn_gain_pct_positive']:.3f}%")
print(f"  Gain mean: 2015={s15['pnn_gain_mean']:.2f}, 2017={s17['pnn_gain_mean']:.2f}")
print(f"  Gain P99: 2015={s15['pnn_gain_p99']:.2f}, 2017={s17['pnn_gain_p99']:.2f}")
print(f"  PNN scores mean: 2015={s15['pnn_score_mean']:.4f}, 2017={s17['pnn_score_mean']:.4f}")
print(f"  PNN scores P99: 2015={s15['pnn_score_p99']:.4f}, 2017={s17['pnn_score_p99']:.4f}")

print("\n--- PRAE ---")
print(f"  n_train: 2015={s15['prae_n_train']}, 2017={s17['prae_n_train']}")
print(f"  Mu mean: 2015={s15['prae_mu_mean']:.6f}, 2017={s17['prae_mu_mean']:.6f}")
print(f"  Mu std: 2015={s15['prae_mu_std']:.6f}, 2017={s17['prae_mu_std']:.6f}")
print(f"  Mu range: 2015=[{s15['prae_mu_min']:.6f},{s15['prae_mu_max']:.6f}], 2017=[{s17['prae_mu_min']:.6f},{s17['prae_mu_max']:.6f}]")
print(f"  Test score mean: 2015={s15['prae_test_score_mean']:.2f}, 2017={s17['prae_test_score_mean']:.2f}")
print(f"  Test score P99: 2015={s15['prae_test_score_p99']:.2f}, 2017={s17['prae_test_score_p99']:.2f}")
print(f"  Test score std: 2015={s15['prae_test_score_std']:.2f}, 2017={s17['prae_test_score_std']:.2f}")

out = os.path.join("figures", "diagnostics", "year_comparison_stats.json")
with open(out, "w") as f:
    json.dump(stats, f, indent=2)
print(f"\nSaved: {out}")
