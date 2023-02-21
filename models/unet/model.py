from torch import nn, cat
from torch.nn.functional import pad
from numpy.typing import NDArray


FILTERS = [64, 128, 256, 512, 1024]


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_classes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes

        # Down sampling/contracting path
        self.input_channel = DoubleConvolution(
            in_channels=in_channels, out_channels=FILTERS[0]
        )
        self.down1 = Down(in_channels=FILTERS[0], out_channels=FILTERS[1])  # 64 -> 128
        self.down2 = Down(in_channels=FILTERS[1], out_channels=FILTERS[2])  # 128 -> 256
        self.down3 = Down(in_channels=FILTERS[2], out_channels=FILTERS[3])  # 256 -> 512
        self.down4 = Down(in_channels=FILTERS[3], out_channels=FILTERS[4])  # 512 -> 1024

        # Up sampling/expansive path
        self.up1 = Up(in_channels=FILTERS[4], out_channels=FILTERS[3])  # 1024 -> 512
        self.up2 = Up(in_channels=FILTERS[3], out_channels=FILTERS[2])  # 512 -> 256
        self.up3 = Up(in_channels=FILTERS[2], out_channels=FILTERS[1])  # 256 -> 128
        self.up4 = Up(in_channels=FILTERS[1], out_channels=FILTERS[0])  # 128 -> 64
        self.out_channel = OutConvolution(
            in_channels=FILTERS[0], out_channels=out_classes
        )

    def forward(self, input):
        x1 = self.input_channel(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out_channel(x)


class Down(nn.Module):
    """
    Downscale:
        - MaxPool
        - DoubleConvolution
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.double_conv = DoubleConvolution(
            in_channels=in_channels, out_channels=out_channels
        )

    def forward(self, inputs):
        outputs = self.maxpool(inputs)
        return self.double_conv(outputs)


class Up(nn.Module):
    """
    Upscale:
        - Transpose Convolution
        - DoubleConvolution
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConvolution(in_channels, out_channels)

    def forward(self, input1: NDArray, input2: NDArray):
        # Inputs are CHW(Channels, Height, Width)
        input1 = self.up(input1)

        diffY = input2.size()[2] - input1.size()[2]
        diffX = input2.size()[3] - input1.size()[3]

        input1 = pad(input1, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])

        return self.conv(cat([input1, input2], dim=1))


class DoubleConvolution(nn.Module):
    """
    (Convolution -> BN -> ReLU) * 2
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        return self.double_conv(inputs)


class OutConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )

    def forward(self, inputs):
        return self.conv(inputs)
