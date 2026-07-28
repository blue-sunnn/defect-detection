import torch.nn as nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

from backend.utils_training import DEVICE


def build_model(num_classes, dropout_rate):
    """Create the pretrained base model and replace the classification head."""
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(128, num_classes),
    )

    return model.to(DEVICE)