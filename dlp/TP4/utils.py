import torch
import torch.nn as nn
from torchvision import models


def precompute_features(
    model: models.ResNet,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
) -> torch.utils.data.Dataset:
    """
    Create a new dataset with the features precomputed by the model.

    If the model is $f \circ g$ where $f$ is the last layer and $g$ is
    the rest of the model, it is not necessary to recompute $g(x)$ at
    each epoch as $g$ is fixed. Hence you can precompute $g(x)$ and
    create a new dataset
    $\mathcal{X}_{\text{train}}' = \{(g(x_n),y_n)\}_{n\leq N_{\text{train}}}$

    Arguments:
    ----------
    model: models.ResNet
        The model used to precompute the features
    dataset: torch.utils.data.Dataset
        The dataset to precompute the features from
    device: torch.device
        The device to use for the computation

    Returns:
    --------
    torch.utils.data.Dataset
        The new dataset with the features precomputed
    """
    model.eval()
    model.to(device)

    # Remove the last layer
    feature_extractor = nn.Sequential(*list(model.children())[:-1])

    features_list = []
    labels_list = []
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = feature_extractor(images)
            features = features.squeeze()
            features_list.append(features.cpu())
            labels_list.append(labels)

    features_tensor = torch.cat(features_list)
    labels_tensor = torch.cat(labels_list)

    return torch.utils.data.TensorDataset(features_tensor, labels_tensor)
