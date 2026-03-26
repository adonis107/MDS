import torch
import torch.nn as nn


class LastLayer(nn.Module):
    """Last layer of the resnet10 model. It takes the features extracted by the resnet10 and outputs the class probabilities."""

    def __init__(self, in_features: int = 512, out_features: int = 2):
        super(LastLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor):
        return self.fc(x)
