import os
import torch

from detection.models.hybrid import TransformerOCSVM
from detection.models.pnn import PNN
from detection.models.prae import PRAE
from detection.models.transformer import BottleneckTransformer
from detection.trainers.callbacks import EarlyStopping
from detection.trainers.training import Trainer


def build_fresh_model(
    model_type, num_features, seq_length, device,
    transformer_cfg, epochs, lr, patience,
    ocsvm_nu, nystroem_components, ocsvm_sgd_lr, ocsvm_sgd_epochs,
    pnn_hidden_dim, prae_sigma,
):
    if model_type == "transformer_ocsvm":
        transformer = BottleneckTransformer(
            num_features=num_features, sequence_length=seq_length, **transformer_cfg)
        early_stop = EarlyStopping(patience=patience, verbose=False,
                                   path='transformer_ocsvm_checkpoint.pth')
        trainer = Trainer(
            epochs=epochs, learning_rate=lr,
            callbacks=[early_stop], device=str(device))
        detector = TransformerOCSVM(
            transformer_model=transformer, trainer=trainer,
            kernel="rbf", nu=ocsvm_nu, gamma="auto",
            n_components=nystroem_components,
            sgd_lr=ocsvm_sgd_lr, sgd_epochs=ocsvm_sgd_epochs)
        return transformer, detector

    if model_type == "pnn":
        input_dim = num_features
        model = PNN(input_dim=input_dim, hidden_dim=pnn_hidden_dim).to(device)
        return model, None

    if model_type == "prae":
        backbone = BottleneckTransformer(
            num_features=num_features, sequence_length=seq_length, **transformer_cfg)
        model = PRAE(backbone_model=backbone, num_train_samples=1,
                     lambda_reg=1.0, sigma=prae_sigma).to(device)
        return model, None

    raise ValueError(f"Unknown model type: {model_type}")


_DEFAULT_TRANSFORMER_CFG = dict(model_dim=128, num_heads=8, num_layers=6, representation_dim=128, dim_feedforward=512)


def load_model(
    model_type, num_features, weights_path, device,
    seq_length=25,
    transformer_cfg=None,
    pnn_hidden_dim=64,
    prae_sigma=0.5,
):
    if transformer_cfg is None:
        transformer_cfg = _DEFAULT_TRANSFORMER_CFG

    if model_type == "transformer_ocsvm":
        transformer = BottleneckTransformer(
            num_features=num_features, sequence_length=seq_length, **transformer_cfg)
        transformer.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True))
        transformer.eval().to(device)
        ocsvm_path = weights_path.replace("_weights.pth", "_detector.pth")
        ocsvm = (
            torch.load(ocsvm_path, map_location=device, weights_only=False)
            if os.path.exists(ocsvm_path) else None
        )
        return transformer, ocsvm

    if model_type == "pnn":
        input_dim = num_features
        model = PNN(input_dim=input_dim, hidden_dim=pnn_hidden_dim).to(device)
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True))
        model.eval()
        return model, None

    if model_type == "prae":
        backbone = BottleneckTransformer(
            num_features=num_features, sequence_length=seq_length, **transformer_cfg)
        model = PRAE(
            backbone_model=backbone, num_train_samples=1,
            lambda_reg=1.0, sigma=prae_sigma).to(device)
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        state_dict = {k: v for k, v in state_dict.items() if k != "mu"}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, None

    raise ValueError(f"Unknown model type: {model_type}")
