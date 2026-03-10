from torch import nn
from torchvision import models


def create_model(num_classes: int = 10) -> nn.Module:
    model = models.resnet18(weights=None)

    # CIFAR-10 images are 32x32, so use a smaller input stem.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
