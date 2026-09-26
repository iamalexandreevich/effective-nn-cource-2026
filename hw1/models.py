from torch import nn


def conv(cin, cout, k, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
        nn.ReLU(inplace=True),
    )


def build_model(num_classes=100):
    return nn.Sequential(
        conv(3, 32, 7, stride=2),
        nn.MaxPool2d(3, stride=2, padding=1),
        conv(32, 64, 5),
        conv(64, 128, 3, stride=2),
        conv(128, 256, 1),
        conv(256, 256, 3, stride=2),
        conv(256, 512, 1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, num_classes),
    )
