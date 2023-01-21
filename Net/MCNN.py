from torch import nn
import torch
from Net.arch_util import *
class MoireCNN(nn.Module):

    def conv_block(self, channels,pre=[]):
        convs = []
        for i in range(5):
            tmp = USConv2d(channels, channels, 3, 1, 1,us=[True,True])
            convs.extend([tmp, nn.ReLU(True)])
            pre = [tmp]
        last_conv = convs[-2]
        last_conv.us = [True,False]
        return nn.Sequential(*convs),last_conv

    def up_conv_block(self, *channels,pre=[]):
        layer_nums = len(channels) - 1
        up_convs = []
        for num in range(layer_nums):
            up_convs += [USConvTranspose2d(channels[num], channels[num + 1],
                                            4, 2, 1,us=[False,False]), nn.ReLU(True)]
        up_convs += [USConv2d(32, 3, 3, 1, 1,us=[False,False])]
        last_conv = up_convs[-1]

        return nn.Sequential(*up_convs),last_conv

    def __init__(self):

        super().__init__()
        
        self.s11 = nn.Sequential()
        s11conv = USConv2d(3, 32, 3, 1, 1,us=[False,True])
        self.s11.add_module('0',s11conv)
        self.s11.add_module('1',nn.ReLU(True))
        s11conv1 = USConv2d(32, 32, 3, 1,1,us=[True,True])
        self.s11.add_module('2',s11conv1)

        self.s13,s13_last = self.conv_block(32)
        self.s12,s12_last = self.up_conv_block()

        self.s21 = nn.Sequential()
        s21conv = USConv2d(32, 32, 3, 2, 1,us=[True,True])
        self.s21.add_module('0',s21conv)
        self.s21.add_module('1',nn.ReLU(True))
        s21conv1 = USConv2d(32, 64, 3, 1, 1,us=[True,True])
        self.s21.add_module('2',s21conv1)
        self.s23, s23_last = self.conv_block(64)
        self.s22,s22_last = self.up_conv_block(64, 32)

        # init_conv = [USConv2d(64, 64, 3, 2, 1), nn.ReLU(True),
        #              USConv2d(64, 64, 3, 1, 1)]
                     
        self.s31 = nn.Sequential()
        s31conv = USConv2d(64, 64, 3, 2, 1,us=[True,True])
        self.s31.add_module('0',s31conv)
        self.s31.add_module('1',nn.ReLU(True))
        s31conv1 = USConv2d(64, 64, 3, 1, 1,us=[True,True])
        self.s31.add_module('2',s31conv1)
        self.s33,s33_last = self.conv_block(64)
        self.s32,s32_last = self.up_conv_block(64, 64, 32)

        # self.s41 = nn.Sequential(*init_conv)

        self.s41 = nn.Sequential()
        s41conv = USConv2d(64, 64, 3, 2, 1,us=[True,True])
        self.s41.add_module('0',s41conv)
        self.s41.add_module('1',nn.ReLU(True))
        s41conv1 = USConv2d(64, 64, 3, 1, 1,us=[True,True])
        self.s41.add_module('2',s41conv1)

        self.s43,s43_last = self.conv_block(64)
        self.s42,s42_last = self.up_conv_block(64, 64, 32, 32)

        self.s51 = nn.Sequential()
        s51conv = USConv2d(64, 64, 3, 2, 1,us=[True,True])
        self.s51.add_module('0',s51conv)
        self.s51.add_module('1',nn.ReLU(True))
        s51conv1 = USConv2d(64, 64, 3, 1, 1,us=[True,True])
        self.s51.add_module('2',s51conv1)

        # self.s51 = nn.Sequential(*init_conv)
        self.s53,s53_last = self.conv_block(64)
        self.s52,s52_last = self.up_conv_block(64, 64, 32, 32, 32)

    def forward(self, x):
        x1 = self.s11(x)
        x2 = self.s21(x1)
        x3 = self.s31(x2)
        x4 = self.s41(x3)
        x5 = self.s51(x4)
        x1 = self.s12(self.s13(x1))
        x2 = self.s22(self.s23(x2))
        x3 = self.s32(self.s33(x3))
        x4 = self.s42(self.s43(x4))
        x5 = self.s52(self.s53(x5))
        x = x1 + x2 + x3 + x4 + x5
        x = torch.sigmoid(x)
        return x
import thop
from thop import profile

if __name__ == '__main__':
    model = MoireCNN()
    img = torch.rand(1,3,1024,1024)
    #model(img)
    macs, params = profile(model, inputs=(img, ))
    print(macs, params)