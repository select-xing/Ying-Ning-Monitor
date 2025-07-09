import torch.nn as nn
from torchvision.models import resnet50
import torch
import torchvision.models as models

class DualAttention(nn.Module):
    """空间-通道双路注意力模块"""

    def __init__(self, in_ch):
        super().__init__()
        self.ch_att = nn.Sequential(  # 通道注意力(网页4原理)
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // 16, 1),  # 压缩比为16:1
            nn.ReLU(),
            nn.Conv2d(in_ch // 16, in_ch, 1),
            nn.Sigmoid()
        )
        self.sp_att = nn.Sequential(  # 空间注意力(网页7结构)
            nn.Conv2d(in_ch, 1, 1),  # 生成空间权重图
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.ch_att(x) + x * self.sp_att(x)


class BottleneckWithAttention(nn.Module):
    """集成注意力机制的残差块"""
    expansion = 4  # 通道扩展系数(网页6参数)

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        # 标准Bottleneck结构(网页2、网页5实现)
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        # 插入双路注意力模块(网页7改进思路)
        self.attention = DualAttention(planes * self.expansion)  # 在残差连接后

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        # 应用注意力机制(网页1、网页4设计)
        out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet50DualAttention(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # 加载ResNet50预训练骨架(网页1、网页3方法)
        base_model = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # 替换标准Bottleneck块(网页6结构)
        self.layer0 = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool
        )
        self.layer1 = self._make_layer(base_model, BottleneckWithAttention, base_model.layer1)
        self.layer2 = self._make_layer(base_model, BottleneckWithAttention, base_model.layer2)
        self.layer3 = self._make_layer(base_model, BottleneckWithAttention, base_model.layer3)
        self.layer4 = self._make_layer(base_model, BottleneckWithAttention, base_model.layer4)

        # 分类头(网页7配置)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BottleneckWithAttention.expansion, num_classes)

    def _make_layer(self, base_model, block, original_layer):
        """重构各阶段残差块"""
        layers = []
        for bottleneck in original_layer.children():
            # 保持原始参数配置(网页5实现逻辑)
            layer = BottleneckWithAttention(
                inplanes=bottleneck.conv1.in_channels,
                planes=bottleneck.conv1.out_channels,
                stride=bottleneck.conv2.stride[0],
                downsample=bottleneck.downsample
            )
            # 参数迁移(保持预训练权重)
            layer.load_state_dict(bottleneck.state_dict(), strict=False)
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)  # [B,64,56,56]
        x = self.layer1(x)  # [B,256,56,56]
        x = self.layer2(x)  # [B,512,28,28]
        x = self.layer3(x)  # [B,1024,14,14]
        x = self.layer4(x)  # [B,2048,7,7]
        x = self.avgpool(x)  # [B,2048,1,1]
        x = torch.flatten(x, 1)  # [B,2048]
        x = self.fc(x)  # [B,num_classes]
        return x