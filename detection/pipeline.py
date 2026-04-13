"""
Anomaly detection pipeline orchestrator.

Coordinates the full workflow -- data loading, feature engineering,
preprocessing, model construction, training, and evaluation -- by
delegating all core logic to the ``detection`` sub-modules.
"""

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from detection.data.datasets import IndexDataset, get_lob
from detection.data.loaders import create_sequences
from detection.data.preprocessing import clean_lob, filter_market_hours
from detection.data.scalers import scaler as QuantileScaler
from detection.data.scalers import EmpiricalBoxCoxScaler
from detection.features.dynamics import compute_dynamics, compute_elasticity
from detection.features.event_flow import compute_event_flow
from detection.features.hawkes import compute_hawkes
from detection.features.imbalance import (
    compute_imbalance,
    compute_weighted_imbalance,
)
from detection.features.ofi import compute_ofi
from detection.features.volatility import compute_volatility
from detection.models.hybrid import TransformerOCSVM
from detection.models.pnn import PNN
from detection.models.prae import PRAE, calculate_heuristic_lambda
from detection.models.transformer import BottleneckTransformer
from detection.thresholds.dspot import DriftStreamingPeakOverThreshold
from detection.thresholds.pot import PeakOverThreshold
from detection.thresholds.rfdr import RollingFalseDiscoveryRate
from detection.thresholds.spot import StreamingPeakOverThreshold
from detection.trainers.callbacks import EarlyStopping
from detection.trainers.training import Trainer

logger = logging.getLogger(__name__)


SCALER_REGISTRY: Dict[str, type] = {
    "minmax": MinMaxScaler,
    "standard": StandardScaler,
    "quantile": QuantileScaler,
    "box_cox": EmpiricalBoxCoxScaler,
}

MODEL_TYPES = {"transformer_ocsvm", "pnn", "prae"}

THRESHOLD_METHODS = {"pot", "spot", "dspot", "rfdr"}



def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return it as a dictionary."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)




class AnomalyDetectionPipeline:
    """Orchestrates the anomaly-detection workflow.

    Every public method corresponds to a distinct stage.  Core logic is
    imported from ``detection.*`` sub-modules; this class only handles
    sequencing, data hand-off, and logging.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self._device: torch.device = self._resolve_device(
            config.get("training", {}).get("device", "auto")
        )

        self.raw_df: Optional[pd.DataFrame] = None
        self.features_df: Optional[pd.DataFrame] = None
        self.feature_names: List[str] = []
        self.scaler: Any = None
        self.model: Any = None
        self.detector: Optional[TransformerOCSVM] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.test_scores: Optional[np.ndarray] = None

        logger.info("Pipeline initialised  --  device: %s", self._device)

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        return torch.device(device_str)

    def load_data(
        self,
        filepath: Optional[str] = None,
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load raw LOB data from disk, clean it, and keep market hours.

        Steps
        -----
        1. Read CSV / Parquet via ``get_lob``.
        2. Drop artifact columns and sort chronologically (``clean_lob``).
        3. Filter to continuous-trading hours (``filter_market_hours``).
        4. Optionally truncate to ``nrows``.

        Parameters
        ----------
        filepath : str, optional
            Overrides ``config.data.filepath``.
        nrows : int, optional
            Overrides ``config.data.nrows``.
        """
        data_cfg = self.config.get("data", {})
        filepath = filepath or data_cfg["filepath"]
        nrows = nrows or data_cfg.get("nrows")

        logger.info("Loading data from %s ...", filepath)
        df = get_lob(filepath)

        df = clean_lob(df)

        time_col: str = data_cfg.get("time_col", "xltime")
        market_open: float = data_cfg.get("market_open_hour", 9.0)
        market_close: float = data_cfg.get("market_close_hour", 17.5)

        df = filter_market_hours(
            df,
            time_col=time_col,
            market_open_hour=market_open,
            market_close_hour=market_close,
        )

        if nrows is not None:
            df = df.head(nrows)

        self.raw_df = df
        logger.info(
            "Loaded %d rows x %d columns.",
            len(self.raw_df),
            len(self.raw_df.columns),
        )
        return self.raw_df

    def engineer_features(
        self,
        df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute all requested feature sets.

        Parameters
        ----------
        df : DataFrame, optional
            Raw LOB DataFrame.  Falls back to ``self.raw_df``.
        """
        df = df if df is not None else self.raw_df
        if df is None:
            raise ValueError("No data loaded.  Call load_data() first.")

        feat_cfg = self.config.get("features", {})
        feature_sets: List[str] = feat_cfg.get(
            "sets", ["base"]
        )
        window: int = feat_cfg.get("window", 50)
        warmup_steps: int = feat_cfg.get("warmup_steps", 3000)
        clip_q: List[float] = feat_cfg.get("clip_quantiles", [0.001, 0.999])

        logger.info("Engineering features: %s", feature_sets)

        df = df.copy()
        features = pd.DataFrame(index=df.index)

        if "base" in feature_sets:
            df = compute_imbalance(df)
            features["L1_Imbalance"] = df["L1_Imbalance"]
            features["L5_Imbalance"] = df["L5_Imbalance"]
            features = compute_dynamics(df, features, window=window)
            features = compute_elasticity(df, features)
            features = compute_volatility(df, features, window=window)

        if "tao" in feature_sets:
            tao_cfgs = feat_cfg.get("tao_configs", [
                {"name": "Weighted_Imbalance_decreasing",
                 "weights": [0.1, 0.1, 0.2, 0.2, 0.4]},
                {"name": "Weighted_Imbalance_increasing",
                 "weights": [0.4, 0.2, 0.2, 0.1, 0.1]},
                {"name": "Weighted_Imbalance_constant",
                 "weights": [0.2, 0.2, 0.2, 0.2, 0.2]},
            ])
            for tc in tao_cfgs:
                features[tc["name"]] = compute_weighted_imbalance(
                    df,
                    weights=tc["weights"],
                    levels=tc.get("levels", 5),
                )

        if "poutre" in feature_sets:
            features = compute_event_flow(df, features)

        if "hawkes" in feature_sets:
            features = compute_hawkes(df, features)

        if "ofi" in feature_sets:
            features = compute_ofi(df, features)

        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features = features.fillna(0)

        if warmup_steps > 0 and len(features) > warmup_steps:
            features = features.iloc[warmup_steps:].reset_index(drop=True)
            if self.raw_df is not None and len(self.raw_df) > warmup_steps:
                self.raw_df = self.raw_df.iloc[warmup_steps:].reset_index(
                    drop=True
                )
            logger.info(
                "Warmup trim: dropped first %d rows for EWMA stabilisation.",
                warmup_steps,
            )
        elif warmup_steps > 0:
            logger.warning(
                "Data length (%d) <= warmup_steps (%d).  Skipping trim.",
                len(features),
                warmup_steps,
            )

        lower = features.quantile(clip_q[0])
        upper = features.quantile(clip_q[1])
        features = features.clip(lower=lower, upper=upper, axis=1)

        std_devs = features.std()
        drop_cols = std_devs[std_devs < 1e-9].index.tolist()
        if drop_cols:
            logger.info(
                "Dropping %d constant/zero-variance columns: %s",
                len(drop_cols),
                drop_cols,
            )
            features = features.drop(columns=drop_cols)

        self.features_df = features
        self.feature_names = features.columns.tolist()
        logger.info(
            "Feature engineering complete.  %d features.", len(self.feature_names)
        )
        return features

    def preprocess(
        self,
        features_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Scale, create sequences, split, and build DataLoaders.

        Parameters
        ----------
        features_df : DataFrame, optional
            Overrides ``self.features_df``.
        """
        features_df = features_df if features_df is not None else self.features_df
        if features_df is None:
            raise ValueError(
                "No features available.  Call engineer_features() first."
            )

        prep_cfg = self.config.get("preprocessing", {})
        scaler_name: str = prep_cfg.get("scaler", "minmax")
        seq_length: int = prep_cfg.get("seq_length", 25)
        target_col: str = prep_cfg.get("target_col", "log_return")
        train_ratio: float = prep_cfg.get("train_ratio", 0.70)
        val_ratio: float = prep_cfg.get("val_ratio", 0.15)
        batch_size: int = self.config.get("training", {}).get("batch_size", 64)
        model_type: str = self.config.get("model", {}).get(
            "type", "transformer_ocsvm"
        )

        if target_col not in features_df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in features."
            )

        logger.info("Scaling with method: %s", scaler_name)
        scaler_cls = SCALER_REGISTRY.get(scaler_name)
        if scaler_cls is None:
            raise ValueError(
                f"Unknown scaler '{scaler_name}'.  "
                f"Choose from {list(SCALER_REGISTRY)}"
            )

        data_values = features_df.values.astype(np.float32)
        n_total = len(data_values) - seq_length
        train_end = int(n_total * train_ratio)
        val_end = int(n_total * (train_ratio + val_ratio))

        train_raw = data_values[: train_end + seq_length]
        val_raw = data_values[train_end : val_end + seq_length]
        test_raw = data_values[val_end :]

        self.scaler = scaler_cls()
        train_scaled = self.scaler.fit_transform(train_raw).astype(np.float32)
        val_scaled = self.scaler.transform(val_raw).astype(np.float32)
        test_scaled = self.scaler.transform(test_raw).astype(np.float32)

        x_train = torch.from_numpy(create_sequences(train_scaled, seq_length))
        x_val = torch.from_numpy(create_sequences(val_scaled, seq_length))
        x_test = torch.from_numpy(create_sequences(test_scaled, seq_length))
        logger.info(
            "Created sequences  --  Train: %d | Val: %d | Test: %d",
            len(x_train), len(x_val), len(x_test),
        )

        target_idx = self.feature_names.index(target_col)

        if model_type == "pnn":
            y_train = torch.from_numpy(
                train_scaled[seq_length:, target_idx]
            ).unsqueeze(1)
            y_val = torch.from_numpy(
                val_scaled[seq_length:, target_idx]
            ).unsqueeze(1)
            y_test = torch.from_numpy(
                test_scaled[seq_length:, target_idx]
            ).unsqueeze(1)

            train_ds = TensorDataset(
                x_train.reshape(x_train.size(0), -1), y_train
            )
            val_ds = TensorDataset(
                x_val.reshape(x_val.size(0), -1), y_val
            )
            test_ds = TensorDataset(
                x_test.reshape(x_test.size(0), -1), y_test
            )

        elif model_type == "prae":
            train_ds = IndexDataset(TensorDataset(x_train, x_train))
            val_ds = IndexDataset(TensorDataset(x_val, x_val))
            test_ds = TensorDataset(x_test, x_test)

        else:
            train_ds = TensorDataset(x_train, x_train)
            val_ds = TensorDataset(x_val, x_val)
            test_ds = TensorDataset(x_test, x_test)

        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False
        )

        logger.info(
            "Data split  --  Train: %d | Val: %d | Test: %d",
            len(train_ds),
            len(val_ds),
            len(test_ds),
        )
        return self.train_loader, self.val_loader, self.test_loader

    def build_model(self) -> Any:
        """Instantiate model (and optional detector) from configuration.

        Uses a factory pattern: the ``model.type`` config key selects which
        builder to invoke.  New model types can be added by extending the
        ``_MODEL_BUILDERS`` mapping.
        """
        model_cfg = self.config.get("model", {})
        model_type: str = model_cfg.get("type", "transformer_ocsvm")
        training_cfg = self.config.get("training", {})
        seq_length: int = self.config.get("preprocessing", {}).get(
            "seq_length", 25
        )
        num_features: int = len(self.feature_names)

        builders = {
            "transformer_ocsvm": self._build_transformer_ocsvm,
            "pnn": self._build_pnn,
            "prae": self._build_prae,
        }

        builder = builders.get(model_type)
        if builder is None:
            raise ValueError(
                f"Unknown model type '{model_type}'.  "
                f"Choose from {list(builders)}"
            )

        logger.info("Building model: %s", model_type)
        builder(model_cfg, training_cfg, seq_length, num_features)
        return self.model


    def _build_transformer_ocsvm(
        self,
        model_cfg: Dict[str, Any],
        training_cfg: Dict[str, Any],
        seq_length: int,
        num_features: int,
    ) -> None:
        t_cfg = model_cfg.get("transformer", {})
        o_cfg = model_cfg.get("ocsvm", {})

        transformer = BottleneckTransformer(
            num_features=num_features,
            model_dim=t_cfg.get("model_dim", 128),
            num_heads=t_cfg.get("num_heads", 8),
            num_layers=t_cfg.get("num_layers", 6),
            representation_dim=t_cfg.get("representation_dim", 128),
            sequence_length=seq_length,
            dim_feedforward=t_cfg.get("dim_feedforward", 512),
        )

        early_stop = EarlyStopping(
            patience=training_cfg.get("patience", 5),
            verbose=True,
        )
        trainer = Trainer(
            epochs=training_cfg.get("epochs", 50),
            learning_rate=training_cfg.get("learning_rate", 1e-3),
            callbacks=[early_stop],
            device=str(self._device),
        )

        self.detector = TransformerOCSVM(
            transformer_model=transformer,
            trainer=trainer,
            kernel=o_cfg.get("kernel", "rbf"),
            nu=o_cfg.get("nu", 0.01),
            gamma=o_cfg.get("gamma", "auto"),
            n_components=o_cfg.get("n_components", 300),
            sgd_lr=o_cfg.get("sgd_lr", 0.01),
            sgd_epochs=o_cfg.get("sgd_epochs", 50),
        )
        self.model = transformer

    def _build_pnn(
        self,
        model_cfg: Dict[str, Any],
        training_cfg: Dict[str, Any],
        seq_length: int,
        num_features: int,
    ) -> None:
        p_cfg = model_cfg.get("pnn", {})
        input_dim = seq_length * num_features

        self.model = PNN(
            input_dim=input_dim,
            hidden_dim=p_cfg.get("hidden_dim", 64),
        ).to(self._device)

    def _build_prae(
        self,
        model_cfg: Dict[str, Any],
        training_cfg: Dict[str, Any],
        seq_length: int,
        num_features: int,
    ) -> None:
        t_cfg = model_cfg.get("transformer", {})
        p_cfg = model_cfg.get("prae", {})

        backbone = BottleneckTransformer(
            num_features=num_features,
            model_dim=t_cfg.get("model_dim", 128),
            num_heads=t_cfg.get("num_heads", 8),
            num_layers=t_cfg.get("num_layers", 6),
            representation_dim=t_cfg.get("representation_dim", 128),
            sequence_length=seq_length,
            dim_feedforward=t_cfg.get("dim_feedforward", 512),
        )

        num_train_samples = (
            len(self.train_loader.dataset) if self.train_loader else 1
        )

        lambda_reg = p_cfg.get("lambda_reg")
        if lambda_reg is None:
            logger.info("Computing heuristic lambda for PRAE ...")
            lambda_reg = calculate_heuristic_lambda(
                self.train_loader, seq_length, num_features
            )
            logger.info("Heuristic lambda: %.6f", lambda_reg)

        self.model = PRAE(
            backbone_model=backbone,
            num_train_samples=num_train_samples,
            lambda_reg=lambda_reg,
            sigma=p_cfg.get("sigma", 0.5),
        ).to(self._device)

    def train(self) -> Dict[str, Any]:
        """Train the current model using the prepared data loaders."""
        model_type: str = self.config.get("model", {}).get(
            "type", "transformer_ocsvm"
        )
        training_cfg = self.config.get("training", {})

        if self.train_loader is None or self.val_loader is None:
            raise ValueError("Data not prepared.  Call preprocess() first.")

        logger.info("Starting training  --  model: %s", model_type)

        train_dispatch = {
            "transformer_ocsvm": self._train_transformer_ocsvm,
            "pnn": self._train_generic,
            "prae": self._train_generic,
        }

        trainer_fn = train_dispatch.get(model_type)
        if trainer_fn is None:
            raise ValueError(f"No trainer registered for '{model_type}'.")

        if model_type == "transformer_ocsvm":
            return trainer_fn()
        return trainer_fn(training_cfg)

    def _train_transformer_ocsvm(self) -> Dict[str, str]:
        self.detector.fit(self.train_loader, self.val_loader)
        logger.info("TransformerOCSVM training complete.")
        return {"status": "trained"}

    def _train_generic(self, training_cfg: Dict[str, Any]) -> Dict[str, str]:
        early_stop = EarlyStopping(
            patience=training_cfg.get("patience", 5),
            verbose=True,
        )
        trainer = Trainer(
            epochs=training_cfg.get("epochs", 50),
            learning_rate=training_cfg.get("learning_rate", 1e-3),
            callbacks=[early_stop],
            device=str(self._device),
        )
        trainer.fit(self.model, self.train_loader, self.val_loader)
        logger.info("Training complete.")
        return {"status": "trained"}

    def evaluate(
        self,
        y_true: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Score the test set, apply a threshold, and report metrics.

        Parameters
        ----------
        y_true : ndarray, optional
            Ground-truth binary labels.  If provided, supervised metrics
            (AUROC, AUPRC, F-beta) are included.
        """
        model_type: str = self.config.get("model", {}).get(
            "type", "transformer_ocsvm"
        )
        beta: int = self.config.get("evaluation", {}).get("beta", 4)

        logger.info("Evaluating model on test set ...")
        scores = self._compute_test_scores(model_type)
        self.test_scores = scores

        threshold = self._find_threshold(scores)
        preds = (scores > threshold).astype(int)

        results: Dict[str, Any] = {
            "threshold": float(threshold),
            "num_anomalies": int(preds.sum()),
            "anomaly_ratio": float(preds.mean()),
        }

        if y_true is not None:
            results.update(
                {
                    "AUROC": float(roc_auc_score(y_true, scores)),
                    "AUPRC": float(average_precision_score(y_true, scores)),
                    "F_beta": float(fbeta_score(y_true, preds, beta=beta)),
                    "confusion_matrix": confusion_matrix(
                        y_true, preds
                    ).tolist(),
                }
            )

        logger.info(
            "Evaluation results: %s",
            {k: v for k, v in results.items() if k != "confusion_matrix"},
        )
        return results


    def _compute_test_scores(self, model_type: str) -> np.ndarray:
        """Return per-sample anomaly scores on the test set."""
        if model_type == "transformer_ocsvm":
            return self.detector.predict(self.test_loader)

        if model_type == "pnn":
            return self._compute_pnn_scores()

        return self._compute_deep_model_scores()

    def _compute_deep_model_scores(self) -> np.ndarray:
        """Per-sample reconstruction error for PRAE / Transformer AE."""
        self.model.eval()
        all_scores: List[np.ndarray] = []
        with torch.no_grad():
            for batch in self.test_loader:
                scores = self.model.get_anomaly_score(batch)
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                all_scores.append(np.atleast_1d(scores))
        return np.concatenate(all_scores)

    def _compute_pnn_scores(self) -> np.ndarray:
        """Per-sample negative log-likelihood for PNN."""
        self.model.eval()
        all_nlls: List[np.ndarray] = []
        with torch.no_grad():
            for batch in self.test_loader:
                x, y = batch
                x = x.to(self._device)
                y = y.to(self._device)
                mu, sigma, alpha = self.model(x)
                z = (y - mu) / sigma
                phi_z = (
                    (1.0 / math.sqrt(2 * math.pi))
                    * torch.exp(-0.5 * z ** 2)
                )
                big_phi = 0.5 * (
                    1 + torch.erf(alpha * z / math.sqrt(2))
                )
                pdf = (2.0 / sigma) * phi_z * big_phi
                nll = -torch.log(pdf + 1e-10)
                all_nlls.append(nll.cpu().numpy().flatten())
        return np.concatenate(all_nlls)


    def _find_threshold(self, scores: np.ndarray) -> float:
        """Select an anomaly threshold using the configured method."""
        t_cfg = self.config.get("threshold", {})
        method: str = t_cfg.get("method", "pot")
        risk: float = t_cfg.get("risk", 1e-4)
        init_level: float = t_cfg.get("init_level", 0.98)
        num_candidates: int = t_cfg.get("num_candidates", 10)

        logger.info("Finding threshold  --  method: %s", method)

        if method == "pot":
            z, _t = PeakOverThreshold(
                scores,
                num_candidates=num_candidates,
                risk=risk,
                init_level=init_level,
            )
            return float(z)

        if method == "spot":
            num_init = t_cfg.get("num_init") or len(scores) // 2
            z_list = StreamingPeakOverThreshold(
                scores,
                num_init=num_init,
                num_candidates=num_candidates,
                risk=risk,
                init_level=init_level,
            )
            return float(z_list[-1]) if z_list else float(
                np.percentile(scores, 99)
            )

        if method == "dspot":
            num_init = t_cfg.get("num_init") or len(scores) // 2
            depth = t_cfg.get("depth", 50)
            thresholds = DriftStreamingPeakOverThreshold(
                scores,
                num_init=num_init,
                depth=depth,
                num_candidates=num_candidates,
                risk=risk,
                init_level=init_level,
            )
            return (
                float(thresholds[-1])
                if len(thresholds) > 0
                else float(np.percentile(scores, 99))
            )

        if method == "rfdr":
            window_size = t_cfg.get("window_size", 500)
            alpha = t_cfg.get("alpha", 0.05)
            rfdr = RollingFalseDiscoveryRate(
                window_size=window_size, alpha=alpha
            )
            threshold = float(np.percentile(scores, 99))
            for s in scores:
                _is_anom, threshold = rfdr.process_new_score(float(s))
            return float(threshold)

        logger.warning(
            "Unknown threshold method '%s'.  Falling back to mean+3*std.",
            method,
        )
        return float(np.mean(scores) + 3.0 * np.std(scores))

    def save(self, output_dir: Optional[str] = None) -> None:
        """Save model weights, configuration, and test scores."""
        out_cfg = self.config.get("output", {})
        output_dir = output_dir or out_cfg.get("results_dir", "results")
        os.makedirs(output_dir, exist_ok=True)

        model_type: str = self.config.get("model", {}).get(
            "type", "transformer_ocsvm"
        )

        if out_cfg.get("save_model", True) and self.model is not None:
            weights_path = os.path.join(
                output_dir, f"{model_type}_weights.pth"
            )
            torch.save(self.model.state_dict(), weights_path)
            logger.info("Model weights saved to %s", weights_path)

        cfg_path = os.path.join(output_dir, "config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as fh:
            yaml.dump(self.config, fh, default_flow_style=False)
        logger.info("Config saved to %s", cfg_path)

        if self.test_scores is not None:
            scores_path = os.path.join(output_dir, "test_scores.npy")
            np.save(scores_path, self.test_scores)
            logger.info("Test scores saved to %s", scores_path)

    def run(self) -> Dict[str, Any]:
        """Execute the complete pipeline end-to-end.

        Sequence: load -> features -> preprocess -> build -> train ->
        evaluate -> save.

        Returns
        -------
        dict
            Evaluation results dictionary.
        """
        self.load_data()
        self.engineer_features()
        self.preprocess()
        self.build_model()
        self.train()
        results = self.evaluate()
        self.save()
        return results
