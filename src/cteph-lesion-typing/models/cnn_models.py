# coding: utf-8

# Imports
import torch
import torch.nn as nn
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50

_MONAI_RESNET_BUILDERS = {10: resnet10, 18: resnet18, 34: resnet34, 50: resnet50}


class Basic3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class SEBlock3D(nn.Module):
    """Channel attention (Squeeze-and-Excitation) for 3D feature maps."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[0], x.shape[1]
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1, 1)
        return x * w


class ResidualBlock3D(nn.Module):
    """3D residual block with optional stride-based downsampling and SE attention."""

    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock3D(out_channels) if use_se else nn.Identity()

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class ResNet3DCNN(nn.Module):
    """ResNet-18-style 3D CNN: residual connections allow deeper training without
    degradation, and SE attention re-weights channels for stronger features."""

    def __init__(self, num_classes=2, dropout=0.3, base_channels=16):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv3d(1, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(c),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(ResidualBlock3D(c, c), ResidualBlock3D(c, c))
        self.layer2 = nn.Sequential(ResidualBlock3D(c, c * 2, stride=2), ResidualBlock3D(c * 2, c * 2))
        self.layer3 = nn.Sequential(ResidualBlock3D(c * 2, c * 4, stride=2), ResidualBlock3D(c * 4, c * 4))
        self.layer4 = nn.Sequential(ResidualBlock3D(c * 4, c * 8, stride=2), ResidualBlock3D(c * 8, c * 8))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c * 8, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class MonaiResNet3D(nn.Module):
    """3D ResNet backbone from MONAI (medicalnet-style, single-channel CT input)."""

    def __init__(self, num_classes=2, depth=18, pretrained=False):
        super().__init__()
        if depth not in _MONAI_RESNET_BUILDERS:
            raise ValueError(f"Unsupported ResNet depth {depth}, choose from {sorted(_MONAI_RESNET_BUILDERS)}")
        self.backbone = _MONAI_RESNET_BUILDERS[depth](
            pretrained=pretrained,
            spatial_dims=3,
            n_input_channels=1,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.backbone(x)


class Deep3DCNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)







