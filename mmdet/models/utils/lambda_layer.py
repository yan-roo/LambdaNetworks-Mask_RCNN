import torch
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, Sequential
from torch import nn as nn

class LambdaBlock(nn.Module):
    def __init__(self, d, dk=16, du=1, Nh=4, m=None, r=23, stride=1):
        super(LambdaBlock, self).__init__()
        self.d = d
        self.dk = dk
        self.du = du
        self.Nh = Nh
        assert d % Nh == 0, 'd should be divided by Nh'
        dv = d // Nh
        self.dv = dv
        assert stride in [1, 2]
        self.stride = stride

        self.conv_qkv = nn.Conv2d(d, Nh*dk + dk*du + dv*du, 1, bias=False)
        self.norm_q = nn.BatchNorm2d(Nh*dk)
        self.norm_v = nn.BatchNorm2d(dv*du)
        self.softmax = nn.Softmax(dim=-1)
        self.lambda_conv = nn.Conv3d(du, dk, (1, r, r), padding = (0, (r-1)// 2, (r-1)// 2))

        if self.stride > 1:
            self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        N, C, H, W = x.shape

        qkv = self.conv_qkv(x)
        q, k, v = torch.split(qkv, [self.Nh * self.dk, self.dk * self.du, self.dv * self.du], dim=1)
        q = self.norm_q(q).view(N, self.Nh, self.dk, H*W)
        v = self.norm_v(v).view(N, self.du, self.dv, H*W)
        k = self.softmax(k.view(N, self.du, self.dk, H*W))

        lambda_c = torch.einsum('bukm,buvm->bkv', k, v)
        yc = torch.einsum('bhkm,bkv->bhvm', q, lambda_c)
        lambda_p = self.lambda_conv(v.view(N, self.du, self.dv, H, W)).view(N, self.dk, self.dv, H*W)
        yp = torch.einsum('bhkm,bkvm->bhvm', q, lambda_p)
        out = (yc + yp).reshape(N, C, H, W)

        if self.stride > 1:
            out = self.avgpool(out)

        return out


class LambdaLayer(Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 downsample_first=True,
                 size=None,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    size=size,
                    **kwargs))
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        size=size,
                        **kwargs))

        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(LambdaLayer, self).__init__(*layers)