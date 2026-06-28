from torchvision.models import efficientnet_b0
import torch.nn as nn

NUM_CLASSES = 4

def create_model():
    model = efficientnet_b0(weights=None)

    num_ftrs = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, NUM_CLASSES)
    )

    return model