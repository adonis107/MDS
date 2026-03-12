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
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import joblib

from detection.data.datasets import IndexDataset
from detection.data.loaders import create_sequences
from detection.data.scalers import scaler as QuantileScaler
from detection.models.transformer import BottleneckTransformer
from detection.models.hybrid import TransformerOCSVM
from detection.models.pnn import PNN
from detection.models.prae import PRAE, calculate_heuristic_lambda, grid_search_lambda
from detection.trainers.training import Trainer
from detection.trainers.callbacks import EarlyStopping

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
os.makedirs(RESULTS_DIR, exist_ok=True)

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
# ## Helper functions
# 
# Reusable helpers for:
# 
# 1. **Data loading** -- reads pre-processed parquet files (xltime + LOB + features).
# 2. **Time-block splitting** -- splits the first hour into alternating 5-min train/val blocks.
# 3. **Scaling and sequencing** -- fits a scaler on training data and builds 3-D sequences.
# 4. **DataLoader construction** -- builds loaders per model type.

# %%
def load_processed(filepath):
    """Load a pre-processed parquet file.

    Returns
    -------
    df : DataFrame   -- xltime + raw LOB columns
    features : DataFrame -- engineered features only
    """
    full = pd.read_parquet(filepath)
    meta_cols = [TIME_COL] + [c for c in LOB_COLUMNS if c in full.columns]
    meta_set = set(meta_cols)
    feat_cols = [c for c in full.columns if c not in meta_set]
    return full[meta_cols], full[feat_cols]


def split_first_hour_blocks(xltime_vals, features):
    """Split the first hour of a day into 5-min train/val blocks.

    Parameters
    ----------
    xltime_vals : np.ndarray  -- xltime values aligned with features
    features : DataFrame

    Returns
    -------
    train_features, val_features : DataFrames
    """
    time_factor = 1.0 / (24.0 * 60.0)  # 1 minute as a fraction of a day
    base_date = np.floor(xltime_vals[0])
    start_time = base_date + MARKET_OPEN_HOUR / 24.0

    train_mask = np.zeros(len(features), dtype=bool)
    val_mask = np.zeros(len(features), dtype=bool)
    block_duration = (TRAIN_BLOCK_MINUTES + VAL_BLOCK_MINUTES) * time_factor
    num_blocks = int(FIRST_HOUR_MINUTES / (TRAIN_BLOCK_MINUTES + VAL_BLOCK_MINUTES))

    for b in range(num_blocks):
        block_start = start_time + b * block_duration
        train_end = block_start + TRAIN_BLOCK_MINUTES * time_factor
        val_end = train_end + VAL_BLOCK_MINUTES * time_factor
        train_mask |= (xltime_vals >= block_start) & (xltime_vals < train_end)
        val_mask |= (xltime_vals >= train_end) & (xltime_vals < val_end)

    train_features = features.loc[train_mask].reset_index(drop=True)
    val_features = features.loc[val_mask].reset_index(drop=True)
    return train_features, val_features


def scale_and_create_loaders(train_feat, val_feat, scaler, model_type, feature_names, fit_scaler=True):
    """Scale features, create sequences, and build DataLoaders.

    Parameters
    ----------
    fit_scaler : bool
        If True, fit the scaler on training data.  If False, use transform only (for incremental days).

    Returns
    -------
    train_loader, val_loader, scaler, feature_names
    """
    if fit_scaler:
        train_scaled = scaler.fit_transform(train_feat.values.astype(np.float32)).astype(np.float32)
    else:
        train_scaled = scaler.transform(train_feat.values.astype(np.float32)).astype(np.float32)

    val_scaled = scaler.transform(val_feat.values.astype(np.float32)).astype(np.float32)

    train_seqs = create_sequences(train_scaled, SEQ_LENGTH)
    val_seqs = create_sequences(val_scaled, SEQ_LENGTH)

    if len(train_seqs) == 0 or len(val_seqs) == 0:
        return None, None, scaler, feature_names

    target_idx = feature_names.index(TARGET_COL)
    train_targets = train_scaled[SEQ_LENGTH:, target_idx][:len(train_seqs)]
    val_targets = val_scaled[SEQ_LENGTH:, target_idx][:len(val_seqs)]

    x_train = torch.tensor(train_seqs, dtype=torch.float32)
    x_val = torch.tensor(val_seqs, dtype=torch.float32)

    if model_type == "pnn":
        y_train = torch.tensor(train_targets, dtype=torch.float32).unsqueeze(1)
        y_val = torch.tensor(val_targets, dtype=torch.float32).unsqueeze(1)
        train_ds = TensorDataset(x_train.reshape(x_train.size(0), -1), y_train)
        val_ds = TensorDataset(x_val.reshape(x_val.size(0), -1), y_val)
    elif model_type == "prae":
        train_ds = IndexDataset(TensorDataset(x_train, x_train))
        val_ds = IndexDataset(TensorDataset(x_val, x_val))
    else:  # transformer_ocsvm
        train_ds = TensorDataset(x_train, x_train)
        val_ds = TensorDataset(x_val, x_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, scaler, feature_names

# %%
def build_fresh_model(model_type, num_features):
    """Factory: create a fresh (untrained) model instance."""
    if model_type == "transformer_ocsvm":
        transformer = BottleneckTransformer(
            num_features=num_features, sequence_length=SEQ_LENGTH, **TRANSFORMER_CFG)
        early_stop = EarlyStopping(patience=PATIENCE, verbose=False)
        trainer = Trainer(epochs=EPOCHS, learning_rate=LR,
                          callbacks=[early_stop], device=str(DEVICE))
        detector = TransformerOCSVM(
            transformer_model=transformer, trainer=trainer,
            kernel="rbf", nu=OCSVM_NU, gamma="auto",
            n_components=NYSTROEM_COMPONENTS,
            sgd_lr=OCSVM_SGD_LR, sgd_epochs=OCSVM_SGD_EPOCHS)
        return transformer, detector

    if model_type == "pnn":
        input_dim = SEQ_LENGTH * num_features
        model = PNN(input_dim=input_dim, hidden_dim=PNN_HIDDEN_DIM).to(DEVICE)
        return model, None

    if model_type == "prae":
        backbone = BottleneckTransformer(
            num_features=num_features, sequence_length=SEQ_LENGTH, **TRANSFORMER_CFG)
        # num_train_samples is set later per block
        model = PRAE(backbone_model=backbone, num_train_samples=1,
                     lambda_reg=1.0, sigma=PRAE_SIGMA).to(DEVICE)
        return model, None

    raise ValueError(f"Unknown model type: {model_type}")


def train_one_block(model, detector, train_loader, val_loader, model_type):
    """Train the model for one 5-minute block (with early stopping)."""
    if model_type == "transformer_ocsvm":
        # Reset early-stopping state so each day starts fresh
        for cb in detector.trainer.callbacks:
            if isinstance(cb, EarlyStopping):
                cb.reset()
        # Only train the transformer autoencoder here;
        # OC-SVM is fitted once after all days
        detector.trainer.fit(detector.transformer, train_loader, val_loader)
        return

    # PNN / PRAE
    early_stop = EarlyStopping(patience=PATIENCE, verbose=False)
    trainer = Trainer(epochs=EPOCHS, learning_rate=LR,
                      callbacks=[early_stop], device=str(DEVICE))
    trainer.fit(model, train_loader, val_loader)

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
    logger.info("=" * 80)
    logger.info("MODEL: %s", model_type.upper())
    logger.info("=" * 80)

    model, detector = None, None
    scaler = QuantileScaler()
    feature_names = None
    scaler_fitted = False
    prae_lambda = None  # will be set via grid search on first usable day

    for day_idx in range(NUM_TRAIN_DAYS):
        filepath = FILES[day_idx]
        day_name = os.path.basename(filepath)
        logger.info("Day %d/%d: %s", day_idx + 1, NUM_TRAIN_DAYS, day_name)

        # Load pre-processed data
        df_day, features = load_processed(filepath)

        if feature_names is None:
            feature_names = features.columns.tolist()

        # Ensure consistent columns across days
        for col in feature_names:
            if col not in features.columns:
                features[col] = 0.0
        features = features[feature_names]

        # Split first hour
        train_feat, val_feat = split_first_hour_blocks(df_day[TIME_COL].values, features)
        if len(train_feat) < SEQ_LENGTH + 1 or len(val_feat) < SEQ_LENGTH + 1:
            logger.warning("Day %d: insufficient data in first hour, skipping.", day_idx + 1)
            continue

        # Scale & create loaders
        fit = not scaler_fitted
        train_loader, val_loader, scaler, feature_names = scale_and_create_loaders(
            train_feat, val_feat, scaler, model_type, feature_names, fit_scaler=fit)
        scaler_fitted = True

        if train_loader is None:
            logger.warning("Day %d: empty loaders after sequencing, skipping.", day_idx + 1)
            continue

        # Build model on first usable day (or update PRAE sample count)
        num_features = len(feature_names)
        if model is None:
            model, detector = build_fresh_model(model_type, num_features)

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

        # Train
        train_one_block(model, detector, train_loader, val_loader, model_type)

        training_log.append({
            "model": model_type,
            "day": day_idx + 1,
            "file": day_name,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
        })

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
                df_day, feats = load_processed(filepath)
                for col in feature_names:
                    if col not in feats.columns:
                        feats[col] = 0.0
                feats = feats[feature_names]
                train_feat, _ = split_first_hour_blocks(df_day[TIME_COL].values, feats)
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

logger.info("All models trained.")


