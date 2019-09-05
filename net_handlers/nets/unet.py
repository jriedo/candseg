import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    File containing a unet with zero padding
"""


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, isdropout, activation=F.relu):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(p=0.2)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.activation = activation
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.isdropout = isdropout

    def forward(self, x):
        x = self.activation(self.conv1(x))
        if self.isdropout:
            x = self.dropout(x)
        x = self.activation(self.conv2(x))
        if self.isdropout:
            x = self.dropout(x)
        x = self.activation(self.conv3(x))
        if self.isdropout:
            x = self.dropout(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, isdropout, activation=F.relu):
        super(UpBlock, self).__init__()
        # a version of trilinear interpolation with in=[N, C, W, H], out=[N, 0.5C, 2W, 2H]
        self.up = lambda input: F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=True)
        self.channel_adjustment = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(p=0.2)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.activation = activation
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.isdropout = isdropout

    def forward(self, x, bridge):
        x = self.up(x)
        x = self.channel_adjustment(x)
        x = torch.cat([x, bridge], 1)
        x = self.activation(self.conv1(x))
        if self.isdropout:
            x = self.dropout(x)
        x = self.activation(self.conv2(x))
        if self.isdropout:
            x = self.dropout(x)
        x = self.activation(self.conv3(x))
        if self.isdropout:
            x = self.dropout(x)
        return x


class UNet(nn.Module):
    def __init__(self, isdropout=False):
        super(UNet, self).__init__()

        self.pool = lambda input: F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=True)

        # downsampling blocks
        self.conv_block1_64 = ConvBlock(1, 64, isdropout=isdropout)
        self.conv_block64_128 = ConvBlock(64, 128, isdropout=isdropout)
        self.conv_block128_256 = ConvBlock(128, 256, isdropout=isdropout)
        self.conv_block256_512 = ConvBlock(256, 512, isdropout=isdropout)
        self.conv_block512_1024 = ConvBlock(512, 1024, isdropout=isdropout)

        # upsampling blocks
        self.up_block1024_512 = UpBlock(1024, 512, isdropout=isdropout)
        self.up_block512_256 = UpBlock(512, 256, isdropout=isdropout)
        self.up_block256_128 = UpBlock(256, 128, isdropout=isdropout)
        self.up_block128_64 = UpBlock(128, 64, isdropout=isdropout)

        # last conv layer  (in CU-Net, exactly here the comes the conditional input)
        self.last = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, x):
        block1 = self.conv_block1_64(x)
        block2 = self.conv_block64_128(self.pool(block1))
        block3 = self.conv_block128_256(self.pool(block2))
        block4 = self.conv_block256_512(self.pool(block3))
        block5 = self.conv_block512_1024(self.pool(block4))

        up1 = self.up_block1024_512(block5, block4)
        up2 = self.up_block512_256(up1, block3)
        up3 = self.up_block256_128(up2, block2)
        up4 = self.up_block128_64(up3, block1)

        out = torch.sigmoid(self.last(up4))
        return out

    def get_feature_map(self, x):
        x = self.conv_block1_64(x)
        return x

    def get_encoded(self, x):
        # used for sampling or feature map inspection
        block1 = self.conv_block1_64(x)
        block2 = self.conv_block64_128(self.pool(block1))
        block3 = self.conv_block128_256(self.pool(block2))
        block4 = self.conv_block256_512(self.pool(block3))
        block5 = self.conv_block512_1024(self.pool(block4))

        return (block1, block2, block3, block4, block5)

    def decode(self, block1, block2, block3, block4, block5, feats=None):
        if feats is not None:
            block1 = feats
        up1 = self.up_block1024_512(block5, block4)
        up2 = self.up_block512_256(up1, block3)
        up3 = self.up_block256_128(up2, block2)
        up4 = self.up_block128_64(up3, block1)

        out = torch.sigmoid(self.last(up4))

        return out