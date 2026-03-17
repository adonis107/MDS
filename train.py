# %% [markdown]
# # Sequential Training
# 
# We train the models sequentially, across multiple days.
# 
# - **Dataset:** TOTF.PA (Euronext Paris), 25 daily LOB snapshots
# - **Scaler:** Quantile (Box-Cox + z-score)
# - **Models:**
#     - Transformer + OC-SVM (hybrid)
#     - PNN (Probabilistic Neural Network)
#     - PRAE (Probabilistic Robust Autoencoder)
# - **Training Strategy:** For each of the first 24 days, use the first hour of market data split into alternating 5-minute blocks of training and validation.
# - **Test Day:** Day 25 (held out entirely).
# - **Features:** base, tao (weighted imbalance), poutre (rapidity / event flow), hawkes (memory), ofi (order flow imbalance).

# %%
import os
import sys
import glob
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import joblib

from detection.data.loaders import create_sequences, load_processed, scale_and_create_loaders
from detection.data.preprocessing import split_first_hour_blocks
from detection.data.scalers import scaler as QuantileScaler
from detection.models.hybrid import TransformerOCSVM
from detection.models.prae import calculate_heuristic_lambda, grid_search_lambda
from detection.trainers.checkpoint import (
    clear_resume_state,
    final_artifacts_exist,
    load_resume_state,
    save_resume_state,
)
from detection.trainers.factory import build_fresh_model
from detection.trainers.training import train_one_block

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("training")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device: %s", DEVICE)

# %% [markdown]
# ## Configuration

# %%
# Paths
DATA_DIR = os.path.join("data", "processed", "TOTF.PA-book")
RESULTS_DIR = os.path.join("results")
RESUME_DIR = os.path.join(RESULTS_DIR, "resume_state")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RESUME_DIR, exist_ok=True)

# Data parameters
TIME_COL = "xltime"
MARKET_OPEN_HOUR = 9.0   # Euronext Paris continuous session
FIRST_HOUR_MINUTES = 60
TRAIN_BLOCK_MINUTES = 5
VAL_BLOCK_MINUTES = 5

# Preprocessing
SEQ_LENGTH = 25
BATCH_SIZE = 64
TARGET_COL = "log_return"

# Training
EPOCHS = 1000
LR = 1e-3
PATIENCE = 20

# Model architectures
TRANSFORMER_CFG = dict(model_dim=64, num_heads=4, num_layers=2, representation_dim=128)
PNN_HIDDEN_DIM = 64
PRAE_SIGMA = 0.5

# OC-SVM (Nyström approximation + linear SGD on CUDA)
OCSVM_NU = 0.01
NYSTROEM_COMPONENTS = 300
OCSVM_SGD_LR = 0.01
OCSVM_SGD_EPOCHS = 500

# LOB columns present in processed files
LOB_COLUMNS = [
    f"{side}-{typ}-{lvl}"
    for lvl in range(1, 11)
    for side, typ in [("bid","price"),("bid","volume"),("ask","price"),("ask","volume")]
]

# File listing
FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
NUM_TRAIN_DAYS = len(FILES) - 3  # Last 3 days are held out for testing
logger.info("Found %d daily files.  Training on %d days, testing on day %d.", len(FILES), NUM_TRAIN_DAYS, len(FILES))


# %% [markdown]
# ## Sequential Training Loop
# 
# For each of the 24 training days and for each model type:
# 
# 1. Load pre-processed parquet (features already computed).
# 2. Split the first hour into 5-min train/val blocks.
# 3. Scale, sequence, and build DataLoaders.
# 4. Train the model on that block (continuing from the previous day's weights).
# 5. After all days, save the final model weights and scaler.

# %%
MODEL_TYPES = ["transformer_ocsvm", "pnn", "prae"]

# Storage for trained artefacts
trained_models = {}   # model_type -> (model, detector_or_None)
trained_scalers = {}  # model_type -> scaler
feature_name_map = {} # model_type -> list of feature names
training_log = []     # list of dicts for summary

for model_type in MODEL_TYPES:
    if final_artifacts_exist(model_type, RESULTS_DIR):
        logger.info("Skipping %s: final artifacts already exist.", model_type)
        continue

    logger.info("=" * 80)
    logger.info("MODEL: %s", model_type.upper())
    logger.info("=" * 80)

    model, detector = None, None
    scaler = QuantileScaler()
    feature_names = None
    scaler_fitted = False
    prae_lambda = None
    start_day = 0

    resume_state = load_resume_state(
        model_type, RESUME_DIR, DEVICE,
        SEQ_LENGTH, TRANSFORMER_CFG, EPOCHS, LR, PATIENCE,
        OCSVM_NU, NYSTROEM_COMPONENTS, OCSVM_SGD_LR, OCSVM_SGD_EPOCHS,
        PNN_HIDDEN_DIM, PRAE_SIGMA,
    )
    if resume_state is not None:
        model = resume_state["model"]
        detector = resume_state["detector"]
        scaler = resume_state["scaler"]
        feature_names = resume_state["feature_names"]
        scaler_fitted = True
        start_day = resume_state["start_day"]
        prae_lambda = resume_state["prae_lambda"]
        logger.info(
            "Resuming %s from day %d/%d.",
            model_type,
            start_day + 1,
            NUM_TRAIN_DAYS,
        )

    for day_idx in range(start_day, NUM_TRAIN_DAYS):
        filepath = FILES[day_idx]
        day_name = os.path.basename(filepath)
        logger.info("Day %d/%d: %s", day_idx + 1, NUM_TRAIN_DAYS, day_name)

        df_day, features = load_processed(filepath, TIME_COL, LOB_COLUMNS)

        if feature_names is None:
            feature_names = features.columns.tolist()

        # Ensure consistent columns across days
        for col in feature_names:
            if col not in features.columns:
                features[col] = 0.0
        features = features[feature_names]

        train_feat, val_feat = split_first_hour_blocks(
            df_day[TIME_COL].values, features,
            MARKET_OPEN_HOUR, FIRST_HOUR_MINUTES, TRAIN_BLOCK_MINUTES, VAL_BLOCK_MINUTES)
        if len(train_feat) < SEQ_LENGTH + 1 or len(val_feat) < SEQ_LENGTH + 1:
            logger.warning("Day %d: insufficient data in first hour, skipping.", day_idx + 1)
            continue

        fit = not scaler_fitted
        train_loader, val_loader, scaler, feature_names = scale_and_create_loaders(
            train_feat, val_feat, scaler, model_type, feature_names,
            SEQ_LENGTH, BATCH_SIZE, TARGET_COL, DEVICE, fit_scaler=fit)
        scaler_fitted = True

        if train_loader is None:
            logger.warning("Day %d: empty loaders after sequencing, skipping.", day_idx + 1)
            continue

        num_features = len(feature_names)
        if model is None:
            model, detector = build_fresh_model(
                model_type, num_features, SEQ_LENGTH, DEVICE,
                TRANSFORMER_CFG, EPOCHS, LR, PATIENCE,
                OCSVM_NU, NYSTROEM_COMPONENTS, OCSVM_SGD_LR, OCSVM_SGD_EPOCHS,
                PNN_HIDDEN_DIM, PRAE_SIGMA,
            )

            # ── PRAE: tune lambda_reg via validation reconstruction loss ──
            if model_type == "prae":
                heuristic_lambda = calculate_heuristic_lambda(
                    train_loader, seq_len=SEQ_LENGTH, num_features=num_features)
                logger.info("PRAE heuristic lambda = %.4f", heuristic_lambda)

                prae_lambda = grid_search_lambda(
                    train_loader, val_loader,
                    heuristic_lambda=heuristic_lambda,
                    num_train_samples=len(train_loader.dataset),
                    num_features=num_features,
                    seq_len=SEQ_LENGTH,
                    device=str(DEVICE),
                    epochs=15,
                    learning_rate=LR,
                    **TRANSFORMER_CFG)
                model.lambda_reg = prae_lambda
                logger.info("PRAE lambda_reg set to %.4f (grid search, validation MSE)", prae_lambda)

        if model_type == "prae":
            n_samples = len(train_loader.dataset)
            model.mu = torch.nn.Parameter(torch.full((n_samples,), 0.5, device=DEVICE))

        train_one_block(model, detector, train_loader, val_loader, model_type,
                        PATIENCE, EPOCHS, LR, DEVICE)

        training_log.append({
            "model": model_type,
            "day": day_idx + 1,
            "file": day_name,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
        })

        save_resume_state(
            model_type=model_type,
            model=model,
            scaler=scaler,
            feature_names=feature_names,
            next_day=day_idx + 1,
            resume_dir=RESUME_DIR,
            prae_lambda=prae_lambda,
        )

    # ----------------------------------------------------------------
    # Fit Nyström OC-SVM once after all days, using the final frozen
    # encoder to re-encode every training day's first-hour data.
    # All latent vectors stay on CUDA -- no numpy round-trip.
    # ----------------------------------------------------------------
    if model_type == "transformer_ocsvm" and detector is not None:
        logger.info("Fitting Nyström OC-SVM on latent representations from all training days...")
        model.eval()
        all_latent = []
        with torch.no_grad():
            for day_idx in range(NUM_TRAIN_DAYS):
                filepath = FILES[day_idx]
                df_day, feats = load_processed(filepath, TIME_COL, LOB_COLUMNS)
                for col in feature_names:
                    if col not in feats.columns:
                        feats[col] = 0.0
                feats = feats[feature_names]
                train_feat, _ = split_first_hour_blocks(
                    df_day[TIME_COL].values, feats,
                    MARKET_OPEN_HOUR, FIRST_HOUR_MINUTES, TRAIN_BLOCK_MINUTES, VAL_BLOCK_MINUTES)
                if len(train_feat) < SEQ_LENGTH + 1:
                    continue
                scaled = scaler.transform(train_feat.values.astype(np.float32)).astype(np.float32)
                seqs = create_sequences(scaled, SEQ_LENGTH)
                if len(seqs) == 0:
                    continue
                # Encode in batches to avoid OOM -- keep on GPU
                x_tensor = torch.tensor(seqs, dtype=torch.float32)
                ds = TensorDataset(x_tensor, x_tensor)
                loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
                for batch in loader:
                    x = batch[0].to(DEVICE)
                    latent = model.get_representation(x)
                    all_latent.append(latent)
        X_latent = torch.cat(all_latent, dim=0)

        # Set gamma via the median heuristic: γ = 1 / median(‖zᵢ − zⱼ‖²)
        # The previous formula (10 / (d*Var)) produced a gamma that made the
        # RBF kernel ≈ 0 between all distinct points (exp(-20)), collapsing
        # Nyström features to zero and yielding a constant decision function.
        gamma = TransformerOCSVM._median_heuristic_gamma(X_latent)
        detector.ocsvm.set_gamma(gamma)
        logger.info("OC-SVM gamma set to %.6f  (median heuristic, d=%d)", gamma, X_latent.shape[1])

        detector.ocsvm.fit(X_latent)
        logger.info("Nyström OC-SVM fitted on %d latent vectors from %d days.",
                     X_latent.shape[0], NUM_TRAIN_DAYS)

    # Save artefacts
    trained_models[model_type] = (model, detector)
    trained_scalers[model_type] = scaler
    feature_name_map[model_type] = feature_names

    # Persist weights
    weights_path = os.path.join(RESULTS_DIR, f"{model_type}_weights.pth")
    torch.save(model.state_dict(), weights_path)
    logger.info("Saved %s weights to %s", model_type, weights_path)

    # Persist the Nyström OC-SVM (PyTorch module) for the hybrid model
    if model_type == "transformer_ocsvm" and detector is not None:
        detector_path = os.path.join(RESULTS_DIR, f"{model_type}_detector.pth")
        torch.save(detector.ocsvm, detector_path)
        logger.info("Saved Nyström OC-SVM detector to %s", detector_path)

    scaler_path = os.path.join(RESULTS_DIR, f"{model_type}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info("Saved scaler to %s", scaler_path)

    # Save feature names
    feat_path = os.path.join(RESULTS_DIR, f"{model_type}_features.txt")
    with open(feat_path, "w") as f:
        f.write("\n".join(feature_names))

    clear_resume_state(model_type, RESUME_DIR)

logger.info("All models trained.")


