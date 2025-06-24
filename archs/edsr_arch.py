import torch.nn as nn
from archs import upsampler
from litsr.utils.registry import ArchRegistry
import math, torch


class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea

class Gated_Conv(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, stride=1, rate=1, activation=nn.ELU()):
        super(Gated_Conv, self).__init__()
        padding = int(rate * (ksize - 1) / 2)
        self.conv = nn.Conv2d(in_ch, 2 * out_ch, kernel_size=ksize, stride=stride, padding=padding, dilation=rate)
        self.activation = activation

    def forward(self, x):
        raw = self.conv(x)
        x1 = raw.split(int(raw.shape[1] / 2), dim=1)
        gate = torch.sigmoid(x1[0])
        out = self.activation(x1[1]) * gate
        return out

class BSConvG(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw=Gated_Conv(
                in_ch=in_channels,
                out_ch=out_channels,
                ksize=1,
                stride=1,
                rate=1,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class BasicBlock(nn.Sequential):
    def __init__(
        self,
        conv,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=False,
        bn=True,
        act=nn.ReLU(True),
    ):

        m = [conv(in_channels, out_channels, kernel_size)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
        self,
        conv,
        n_feats,
        kernel_size,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):

        super(ResBlock, self).__init__()
        # m = []
        # for i in range(2):
        #     m.append(conv(n_feats, n_feats, kernel_size))
        #     if bn:
        #         m.append(nn.BatchNorm2d(n_feats))
        #     if i == 0:
        #         m.append(act)

        # self.body = nn.Sequential(*m)

        self.conv1 = conv(n_feats, n_feats, 3)
        self.relu = nn.ReLU()
        self.conv2 = conv(n_feats, n_feats, 3)
        self.res_scale = res_scale

    def forward(self, x):
        # res = self.body(x).mul(self.res_scale)
        res = self.conv2(self.relu(self.conv1(x)))
        res = res * self.res_scale
        res = res + x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)



@ArchRegistry.register()
class EDSR_V(nn.Module):
    def __init__(self, scale, conv=default_conv):
        super(EDSR_V, self).__init__()

        n_resblocks = 16
        n_feats = 64
        kernel_size = 3 
        scale = scale
        act = nn.ReLU(True)

        # define head module
        m_head = [default_conv(3, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(default_conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(default_conv, scale, n_feats, act=False),
            default_conv(n_feats, 3, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x 
    
class ResBlock_IDL(nn.Module):
    def __init__(
        self,
        conv,
        n_feats,
        kernel_size,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):

        super(ResBlock_IDL, self).__init__()

        self.conv1 = conv(n_feats, n_feats, 3)
        self.relu = nn.ReLU()
        self.conv2 = conv(n_feats, n_feats, 3)
        self.res_scale = res_scale

    def forward(self, x):
        # res = self.body(x).mul(self.res_scale)
        res = self.conv2(self.relu(self.conv1(x)))
        res = res * self.res_scale
        res = res - x
        return res



@ArchRegistry.register()
class EDSR_IDL(nn.Module):
    def __init__(self, scale, conv=default_conv):
        super(EDSR_IDL, self).__init__()

        n_resblocks = 16
        n_feats = 64
        kernel_size = 3 
        scale = scale
        act = nn.ReLU(True)

        # define head module
        m_head = [default_conv(3, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock_IDL(
                conv, n_feats, kernel_size, act=act, res_scale=1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(default_conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(default_conv, scale, n_feats, act=False),
            default_conv(n_feats, 3, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x 
