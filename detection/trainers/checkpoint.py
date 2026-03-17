import json
import os

import joblib
import torch

from detection.trainers.factory import build_fresh_model


def final_artifacts_exist(model_type, results_dir):
    required = [
        os.path.join(results_dir, f"{model_type}_weights.pth"),
        os.path.join(results_dir, f"{model_type}_scaler.pkl"),
        os.path.join(results_dir, f"{model_type}_features.txt"),
    ]
    if model_type == "transformer_ocsvm":
        required.append(os.path.join(results_dir, f"{model_type}_detector.pth"))
    return all(os.path.exists(path) for path in required)


def get_resume_paths(model_type, resume_dir):
    return {
        "meta": os.path.join(resume_dir, f"{model_type}_meta.json"),
        "weights": os.path.join(resume_dir, f"{model_type}_weights.pth"),
        "scaler": os.path.join(resume_dir, f"{model_type}_scaler.pkl"),
    }


def save_resume_state(model_type, model, scaler, feature_names, next_day,
                      resume_dir, prae_lambda=None):
    paths = get_resume_paths(model_type, resume_dir)

    model_state = model.state_dict()
    if model_type == "prae" and "mu" in model_state:
        model_state = {k: v for k, v in model_state.items() if k != "mu"}

    torch.save(model_state, paths["weights"])
    joblib.dump(scaler, paths["scaler"])

    payload = {
        "next_day": int(next_day),
        "feature_names": feature_names,
        "prae_lambda": None if prae_lambda is None else float(prae_lambda),
    }
    with open(paths["meta"], "w") as f:
        json.dump(payload, f)


def load_resume_state(
    model_type, resume_dir, device,
    seq_length, transformer_cfg, epochs, lr, patience,
    ocsvm_nu, nystroem_components, ocsvm_sgd_lr, ocsvm_sgd_epochs,
    pnn_hidden_dim, prae_sigma,
):
    paths = get_resume_paths(model_type, resume_dir)
    if not all(os.path.exists(paths[k]) for k in ["meta", "weights", "scaler"]):
        return None

    with open(paths["meta"], "r") as f:
        meta = json.load(f)

    feature_names = meta.get("feature_names")
    next_day = int(meta.get("next_day", 0))
    if not feature_names or next_day <= 0:
        return None

    model, detector = build_fresh_model(
        model_type, len(feature_names), seq_length, device,
        transformer_cfg, epochs, lr, patience,
        ocsvm_nu, nystroem_components, ocsvm_sgd_lr, ocsvm_sgd_epochs,
        pnn_hidden_dim, prae_sigma,
    )
    state_dict = torch.load(paths["weights"], map_location=device)
    if model_type == "prae":
        model.load_state_dict(state_dict, strict=False)
        if meta.get("prae_lambda") is not None:
            model.lambda_reg = float(meta["prae_lambda"])
    else:
        model.load_state_dict(state_dict)

    scaler = joblib.load(paths["scaler"])
    return {
        "model": model,
        "detector": detector,
        "scaler": scaler,
        "feature_names": feature_names,
        "start_day": next_day,
        "prae_lambda": meta.get("prae_lambda"),
    }


def clear_resume_state(model_type, resume_dir):
    paths = get_resume_paths(model_type, resume_dir)
    for path in paths.values():
        if os.path.exists(path):
            os.remove(path)
