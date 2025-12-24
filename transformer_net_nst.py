"""
Transformer Network for Fast Neural Style Transfer

Based on Johnson et al. "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
Architecture: 3 downsampling conv layers -> 5 residual blocks -> 3 upsampling layers
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block with instance normalization and ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return out + residual


class UpsampleBlock(nn.Module):
    """Upsampling block using transposed convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class TransformerNet(nn.Module):
    """
    Image Transformation Network for Neural Style Transfer.

    Takes an input image and outputs a stylized version.
    The network is fully convolutional and can process images of any size.
    """

    def __init__(self):
        super().__init__()

        # Initial convolution (reflection padding for better borders)
        self.pad = nn.ReflectionPad2d(40)

        # Downsampling layers
        self.down1 = ConvBlock(3, 32, kernel_size=9, stride=1)
        self.down2 = ConvBlock(32, 64, kernel_size=3, stride=2)
        self.down3 = ConvBlock(64, 128, kernel_size=3, stride=2)

        # Residual blocks
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # Upsampling layers
        self.up1 = UpsampleBlock(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = UpsampleBlock(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Final output layer
        self.final = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        # Store original size for cropping
        _, _, h, w = x.shape

        # Reflection padding to reduce border artifacts
        x = self.pad(x)

        # Downsampling
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        # Residual blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        # Upsampling
        x = self.up1(x)
        x = self.up2(x)

        # Final convolution (no activation - matches PyTorch reference)
        x = self.final(x)

        # Crop to original size (remove padding effect)
        _, _, out_h, out_w = x.shape
        crop_h = (out_h - h) // 2
        crop_w = (out_w - w) // 2
        x = x[:, :, crop_h:crop_h+h, crop_w:crop_w+w]

        return x


if __name__ == "__main__":
    # Test the network
    model = TransformerNet()
    print(f"TransformerNet parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with a sample input
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
