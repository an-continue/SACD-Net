from typing import Dict
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(Decoder, self).__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)

class SurfaceConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(SurfaceConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        """
         初始化空间注意力模块
         Args:
             kernel_size (int): 卷积核大小，通常为7x7
         """
        super().__init__()
        # 确保kernel_size是奇数，以便padding
        assert kernel_size % 2 == 1
        padding = kernel_size // 2

        self.sigmoid = nn.Sigmoid()

        # 定义7x7卷积层，输入通道为2（平均池化和最大池化的结果），输出通道为1
        self.conv = nn.Conv2d(
            in_channels=2,  # 输入通道数为2（平均池化和最大池化的结果）
            out_channels=1,  # 输出通道数为1（生成空间注意力图）
            kernel_size=kernel_size,  # 卷积核大小，通常为7x7
            padding=padding,  # 填充，保持特征图大小不变
            bias=False  # 不使用偏置
        )

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # F_avg^s [B,1,H,W]
        # 注意这里返回值是两个，最大值和索引，要用两个参数接
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # F_max^s [B,1,H,W]
        # 拼接平均池化和最大池化的结果
        pooled_features = torch.cat((avg_pool.sigmoid(), max_pool.sigmoid()), dim=1)  # [B,2,H,W]
        # 通过 7 * 7 卷积层处理
        spatial_attention = self.conv(pooled_features)
        # # sigmoid激活
        # spatial_attention = self.sigmoid(spatial_attention)
        return x * spatial_attention
class SACBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SACBlock, self).__init__()
        self.Surface=SurfaceConv(in_channels,in_channels)
        # 点卷积
        self.point = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # 线卷积
        self.line1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
        self.line2= nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels*3, out_channels, kernel_size=3, stride=1,padding=1, bias=False)

        self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
        # 宽度池化（压缩高度为1，使用最大池化）
        self.pool_w = nn.AdaptiveMaxPool2d((1, None))
        self.pool_ = nn.AdaptiveMaxPool2d((1, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
    def forward(self, x):
        # 点操作
        b, c, h, w = x.size()
        x_point = self.pool_(x).reshape(b, c, 1, 1).sigmoid()
        # 逐元素相乘
        x_point = x_point * x
        x_point = self.point(x_point)
        #面操作
        x_block = self.Surface(x)
        # 线操作
        h_pooled = self.pool_h(x)  # 形状: (batch_size, channels, height, 1)
        w_pooled = self.pool_w(h_pooled)  # 形状: (batch_size, channels, 1, width)
        x_line = torch.mul(h_pooled.sigmoid(), w_pooled.sigmoid())  # 广播相乘
        x_line=x_line+self.line1(x)+self.line2(x)
        x=torch.cat([x_line,x_point,x_block],dim=1)
        x=self.conv2(x)
        x=self.bn(x)
        x=self.relu(x)
        return x

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            SACBlock(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = Decoder(in_channels, out_channels, in_channels // 2)
            self.atten=SpatialAttention(3)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = Decoder(in_channels, out_channels)
            self.atten = SpatialAttention(3)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        #print(x2.shape)
        x2 = self.atten(x2)
        x1 = self.atten(x1)

        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class SACNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 32):
        super(SACNet, self).__init__()

        import torchvision
        pretrained = torchvision.models.resnet18(pretrained=True)
        pretrained.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,  # 调整步长为 1
            padding=1,
            bias=False
        )
        # 复制预训练权重到新 conv1（如果需要预训练权重）
        if pretrained.conv1.weight.data.shape == pretrained.conv1.weight.data.shape:
            pretrained.conv1.weight.data = pretrained.conv1.weight.data.clone()

        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = Decoder(3, 3)
        #self.in_conv2 = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c * 2, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)


        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 3, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)
        self.out_conv2 = OutConv(base_c*2, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b1 = self.relu(self.bn1(self.conv1(x)))#torch.Size([2, 64, 128, 128])
        #b2 = self.maxpool(b1)
        b1 = self.layer1(b1) #torch.Size([2, 64, 64, 64])
        x2 = self.down1(b1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, b1)
        logits1 = self.out_conv(x)
        return {"out": logits1}

net = SACNet()
summary(net, input_size=(3, 128, 128), device='cpu')  #打印网络结构