"""
https://github.com/dongzhuoyao/3D-ResNets-PyTorch/blob/master/models/resnet.py
"""

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

__all__ = [
    "ResNet",
    "resnet10",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnet200",
]


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
    ).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        sample_size,
        sample_duration,
        shortcut_type="B",
        num_classes=400,
        in_channels=3,
    ):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)

        # Adaptive pooling instead of fixed-size pooling for flexible input sizes
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append(f"layer{i}")
    ft_module_names.append("fc")

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({"params": v})
                break
        else:
            parameters.append({"params": v, "lr": 0.0})

    return parameters


def resnet10(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


# ================================================================
# Medical Imaging Wrapper for 3D Brain MRI
# ================================================================
class ResNet3DMedical(nn.Module):
    """
    Wrapper for ResNet adapted to 3D medical imaging pipeline.

    Converts IMG_SIZE format [H, W, D] to sample_size and sample_duration
    expected by the original ResNet implementation.

    Args:
        img_size: Tuple of (height, width, depth) for input images
        num_classes: Number of output classes
        in_channels: Number of input channels (1 for grayscale MRI)
        resnet_variant: ResNet variant function (resnet10, resnet18, etc.)
        shortcut_type: Type of shortcut connection ('A' or 'B')
        dropout: Dropout rate (applied before final classifier)
    """

    def __init__(
        self,
        img_size=(144, 168, 144),
        num_classes=5,
        in_channels=1,
        resnet_variant="resnet18",
        shortcut_type="B",
        dropout=0.0,
    ):
        super().__init__()

        # Convert img_size [H, W, D] to ResNet's expected format
        # Original ResNet expects: sample_size (spatial), sample_duration (temporal)
        # For 3D MRI: use average of H,W as sample_size, D as sample_duration
        height, width, depth = img_size
        sample_size = int((height + width) / 2)
        sample_duration = depth

        # Build ResNet model
        resnet_func = globals()[resnet_variant]
        self.resnet = resnet_func(
            sample_size=sample_size,
            sample_duration=sample_duration,
            shortcut_type=shortcut_type,
            num_classes=num_classes,
            in_channels=in_channels,
        )

        # Add dropout before final classifier if specified
        if dropout > 0:
            # Replace the final FC layer with dropout + FC
            in_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Sequential(
                nn.Dropout(p=dropout), nn.Linear(in_features, num_classes)
            )

    def forward(self, x):
        return self.resnet(x)


# Convenience functions for medical imaging variants
def resnet10_medical(**kwargs):
    """ResNet-10 for 3D medical imaging"""
    return ResNet3DMedical(resnet_variant="resnet10", **kwargs)


def resnet18_medical(**kwargs):
    """ResNet-18 for 3D medical imaging"""
    return ResNet3DMedical(resnet_variant="resnet18", **kwargs)


def resnet34_medical(**kwargs):
    """ResNet-34 for 3D medical imaging"""
    return ResNet3DMedical(resnet_variant="resnet34", **kwargs)


def resnet50_medical(**kwargs):
    """ResNet-50 for 3D medical imaging"""
    return ResNet3DMedical(resnet_variant="resnet50", **kwargs)


def resnet101_medical(**kwargs):
    """ResNet-101 for 3D medical imaging"""
    return ResNet3DMedical(resnet_variant="resnet101", **kwargs)


def resnet152_medical(**kwargs):
    """ResNet-152 for 3D medical imaging"""
    return ResNet3DMedical(resnet_variant="resnet152", **kwargs)


def resnet200_medical(**kwargs):
    """ResNet-200 for 3D medical imaging"""
    return ResNet3DMedical(resnet_variant="resnet200", **kwargs)
