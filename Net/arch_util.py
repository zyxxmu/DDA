import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np
from torch import Tensor
from typing import Optional, List
import pdb

def make_divisible(v, divisor=8, min_value=8):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        # self.requires_grad = False
        for p in self.parameters():
            p.requires_grad = False

class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, us=[False, False],cat_shape=None,set_cat_mul=None):
        super(USConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.width_mult = 1
        self.us = us
        self.cat_shape = cat_shape
        self.set_cat_mul = set_cat_mul
        self.in_channels_index_list = [None]*4

        self.unrank = False
        # self.pre = pre

    def forward(self, inputs):
        # self.unrank=True
        in_channels = inputs.shape[1] // self.groups if self.us[0] else self.in_channels // self.groups
        out_channels = int(self.out_channels * self.width_mult) if self.us[1] else self.out_channels
        if self.width_mult < 0.3:
            in_channels_index=self.in_channels_index_list[-1]
        elif self.width_mult < 0.6:
            in_channels_index=self.in_channels_index_list[-2]
        elif self.width_mult < 0.8:
            in_channels_index=self.in_channels_index_list[-3]
        else:
            in_channels_index=self.in_channels_index_list[-4]
        
        if in_channels == self.in_channels:
            weight = self.weight[:out_channels, :in_channels, :, :]
        elif in_channels_index is None and self.cat_shape is not None:                                              # concat的时候需要分段取
            if self.set_cat_mul is None:
                cat_num = len(self.cat_shape)                  
                inchannel_index = np.zeros(self.in_channels)
                start = 0
                for i in range(cat_num):
                    inchannel_index[start:start+int(self.width_mult*self.cat_shape[i])]=1
                    start += self.cat_shape[i]
            else:
                assert len(self.set_cat_mul) == len(self.cat_shape), 'USconv2d use cat now and partially prune, need len(self.set_cat_mul) == len(self.cat_shape)'
                inchannel_index = np.zeros(self.in_channels)
                start=0
                for i in range(len(self.set_cat_mul)):
                    if self.set_cat_mul[i] == True:
                        inchannel_index[start:start+int(self.width_mult*self.cat_shape[i])]=1
                    else:
                        inchannel_index[start:start+int(self.cat_shape[i])]=1
                    start += self.cat_shape[i]
            # pdb.set_trace()
            inchannel_index = np.squeeze(np.argwhere(inchannel_index))
            in_channels_index = inchannel_index
            weight = self.weight[:out_channels,inchannel_index, :, :]
        elif in_channels_index is not None and self.cat_shape is not None:
            inchannel_index = in_channels_index
            weight = self.weight[:out_channels,inchannel_index, :, :]
        else:
            weight = self.weight[:out_channels, :in_channels, :, :]

        if self.bias is not None:
            bias = self.bias[:out_channels]
        else:
            bias = self.bias
        y = F.conv2d(inputs, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        # self.y = y
        return y


class USConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, us=[False, False]):
        super(USConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding = output_padding)
        self.width_mult = None
        self.us = us

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        # in_channels = make_divisible(self.in_channels * self.width_mult) if self.us[0] else self.in_channels
        in_channels = int(self.in_channels * self.width_mult) if self.us[0] else self.in_channels
        out_channels = input.shape[1] if self.us[1] else self.out_channels


        weight = self.weight[:in_channels, :out_channels, :, :]

        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)  # type: ignore[arg-type]

        return F.conv_transpose2d(input, weight, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)


class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, width_list = None):
        super(USBatchNorm2d, self).__init__(num_features, affine=True, track_running_stats=False)
        self.width_id = None

        self.bn = nn.ModuleList([
            nn.BatchNorm2d(self.num_features, affine=False) for _ in range(len(width_list))
        ])
        # raise NotImplementedError

    def forward(self, inputs):
        num_features = inputs.size(1)
        y = F.batch_norm(
                inputs,
                self.bn[self.width_id].running_mean[:num_features],
                self.bn[self.width_id].running_var[:num_features],
                self.weight[:num_features],
                self.bias[:num_features],
                self.training,
                self.momentum,
                self.eps)
        return y
