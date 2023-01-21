import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import cos, pi, sqrt
from Net.arch_util import *

import pdb

class ScaleLayer2(nn.Module):
    def __init__(self):
        super().__init__()
        self.ReLU = nn.ReLU()
        self.it_weights = nn.Parameter(torch.autograd.Variable(torch.ones((64, 1, 1, 1)),
                                                               requires_grad=True))

    def forward(self, input):
        y = input.to('cuda') * self.it_weights
        # print(self.it_weights[0,0,0,0])
        return self.ReLU(y)

    def compute_output_shape(self, input_shape):
        return input_shape

#
class Kernel(nn.Module):
    def __init__(self):
        super().__init__()
        conv_shape = (64, 64, 1, 1)
        kernel = torch.zeros(conv_shape).cuda()
        r1 = sqrt(1.0 / 8)
        r2 = sqrt(2.0 / 8)
        for i in range(8):
            _u = 2 * i + 1
            for j in range(8):
                _v = 2 * j + 1
                index = i * 8 + j
                for u in range(8):
                    for v in range(8):
                        index2 = u * 8 + v
                        t = cos(_u * u * pi / 16) * cos(_v * v * pi / 16)
                        t = t * r1 if u == 0 else t * r2
                        t = t * r1 if v == 0 else t * r2
                        kernel[index2, index, 0, 0] = t
        # kernel = torch.from_numpy(kernel)
        self.kernel = torch.autograd.Variable(kernel)

    def forward(self):
        # print(self.kernel)
        return self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape


class adaptive_implicit_trans(nn.Module):
    def __init__(self):
        super().__init__()
        self.ReLU = nn.ReLU()
        self.it_weights = ScaleLayer2()
        self.kernel = Kernel()

    def forward(self, inputs):
        self.kernel1 = self.it_weights(self.kernel())     # 출력도 kernel사이즈랑 동일
        # print(self.kernel1)

        # self.it_weights = self.it_weights.type(torch.cuda.FloatTensor)
        # self.kernel = self.kernel.type(torch.cuda.FloatTensor)
        y = F.conv2d(inputs, self.kernel1)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

class ScaleLayer(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.kernel = nn.Parameter( torch.tensor( s ) )

    def forward(self, input):
        y = input * self.kernel
        return y

    def compute_output_shape(self, input_shape):
        return input_shape


class conv_relu1(nn.Module):
    def __init__(self, channel, filters, kernel, padding=1, stride = 1, set_cat_mul=None,us=[True,True], cat_shape=None):
        super().__init__()
        self.channel = channel
        self.filters = filters
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.conv = USConv2d(self.channel, self.filters, self.kernel, stride=self.stride, padding=self.padding, us=us,cat_shape=cat_shape, set_cat_mul=set_cat_mul)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        return y

class conv_relu_in_block(nn.Module):
    def __init__(self, channel, filters, kernel, padding=1, stride = 1, dilation = 1, us=[True, True],cat_shape=None,set_cat_mul=None):
        super().__init__()
        self.channel = channel
        self.filters = filters
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.relu = nn.ReLU()
        if kernel ==1 :
            #self.conv = nn.Conv2d(self.channel, self.filters, self.kernel, padding=0 )
            self.conv = USConv2d(self.channel, self.filters, self.kernel, stride=self.stride, padding=0, us=us,cat_shape=cat_shape, set_cat_mul=set_cat_mul)
        elif self.stride ==2:
            #self.conv = nn.Conv2d(self.channel, self.filters, self.kernel, padding=1, stride=self.stride)
            self.conv = USConv2d(self.channel, self.filters, self.kernel, stride=self.stride, padding=1, us=us,cat_shape=cat_shape,set_cat_mul=set_cat_mul)
            # self.conv = nn.Conv2d(self.channel, self.filters, self.kernel, padding='same', stride=self.stride)
        elif self.dilation ==1:
            #self.conv = nn.Conv2d(self.channel, self.filters, self.kernel, padding =1, dilation =self.dilation)
            self.conv = USConv2d(self.channel, self.filters, self.kernel, stride=self.stride, padding=1, dilation=self.dilation, us=us, cat_shape=cat_shape,set_cat_mul=set_cat_mul)
        elif self.dilation ==2:
            padding = self.dilation
            #self.conv = nn.Conv2d(self.channel, self.filters, self.kernel, padding=padding, dilation=self.dilation)
            self.conv = USConv2d(self.channel, self.filters, self.kernel, stride=self.stride, padding=padding, dilation=self.dilation, us=us, cat_shape=cat_shape,set_cat_mul=set_cat_mul)
        elif self.dilation ==3:
            padding = self.dilation
            #self.conv = nn.Conv2d(self.channel, self.filters, self.kernel, padding=padding, dilation =self.dilation)
            self.conv = USConv2d(self.channel, self.filters, self.kernel, stride=self.stride, padding=padding, dilation =self.dilation, us=us, cat_shape=cat_shape,set_cat_mul=set_cat_mul)

    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        return y

        # self.last_conv = self.conv

class conv1(nn.Module):
    def __init__(self, channel, filters, kernel, padding = 1, strides=1, us=[True, True], cat_shape=None,set_cat_mul=None):
        super().__init__()
        self.channel = channel
        self.filters = filters
        self.kernel = kernel
        self.padding = padding

        self.conv = USConv2d(self.channel, self.filters, self.kernel, stride=strides, padding=self.padding, us=us, cat_shape=cat_shape, set_cat_mul=set_cat_mul)

        # self.last_conv = self.conv
    def forward(self, x):
        y = self.conv(x)
        return y



class pre_block(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.nFilters = 64
        self.dilation = dilation
        self.add=True
        cat_shape = [128]
        self.conv_relu1 = conv_relu_in_block(128, self.nFilters, 3, padding = 1, dilation = self.dilation[0],cat_shape=cat_shape)
        cat_shape2=[64]+cat_shape
        # pre2=pre+[self.conv_relu1.conv]
        self.conv_relu2 = conv_relu_in_block(192, self.nFilters, 3, padding = 1, dilation = self.dilation[1], cat_shape=cat_shape2)
        cat_shape3=[64]+cat_shape2
        # pre3=pre2+[self.conv_relu2.conv]
        self.conv_relu3 = conv_relu_in_block(256, self.nFilters, 3, padding = 1, dilation = self.dilation[2], cat_shape=cat_shape3)
        cat_shape4=[64]+cat_shape3
        # pre4=pre3+[self.conv_relu3.conv]
        self.conv_relu4 = conv_relu_in_block(320, self.nFilters, 3, padding = 1, dilation = self.dilation[3], cat_shape=cat_shape4)
        cat_shape5=[64]+cat_shape4
        # pre5=pre4+[self.conv_relu4.conv]
        self.conv_relu5 = conv_relu_in_block(384, self.nFilters, 3, padding = 1, dilation = self.dilation[4], cat_shape=cat_shape5)

        cat_shape6=[64]+cat_shape5
        self.conv1 = conv1(448,self.nFilters,3,padding=1, us=[True, False], cat_shape=cat_shape6)
        self.conv2 = conv1(64,self.nFilters*2,1,padding=0, us=[False, True])

        # self.last_conv=self.conv2.conv

        self.relu = nn.ReLU()
        self.adaptive_implicit_trans1 = adaptive_implicit_trans()# 여기 바꾸기
        self.ScaleLayer1 = ScaleLayer(0.1) # 이거해야함

    def forward(self, x):
        # pdb.set_trace()
        t = x
        _t = self.conv_relu1(t)
        #print('\n\nconv_relu11_input',t.shape)  #128,512,512
        #print('conv_relu11_output',_t.shape)    #64 ,512,512
        t = torch.cat( [_t, t ] , dim=-3)
        _t = self.conv_relu2(t)
        #print('\n\nconv_relu22_input',t.shape) #192,512,512
        #print('conv_relu22_output',_t.shape)    #64,512,512
        t = torch.cat( [_t, t ] , dim=-3)
        _t = self.conv_relu3(t)
        #print('conv_relu33_input',t.shape)     #256,512,512
        #print('conv_relu33_output',_t.shape)    #64,512,512
        t = torch.cat( [_t, t ] , dim=-3)
        _t = self.conv_relu4(t)
        #print('conv_relu44_input',t.shape)     #320,512,512
        #print('conv_relu44_output',_t.shape)    #64,512,512
        t = torch.cat( [_t, t ] , dim=-3)
        _t = self.conv_relu5(t)
        #print('conv_relu55_input',t.shape)     #384,512,512
        #print('conv_relu55_output',_t.shape)    #64,512,512
        t = torch.cat( [_t, t ] , dim=-3)
        #print('t',t.shape)                      #448,512,512
        t = self.conv1(t)
        #print('conv1_output',t.shape)           #64,512,512

        t = self.adaptive_implicit_trans1(t)
        t = self.conv2(t)
        # print('t77',t.shape)
        t = self.ScaleLayer1(t)

        t = torch.add( x,t )
        # y = self.relu(t)
        return t




class global_block(nn.Module):
    def __init__(self,x):
        super().__init__()
        self.size = (x+2) // 2
        self.avgkernel_size = x
        self.nFilters = 64
        self.conv_func1 = conv_relu_in_block(128, self.nFilters*4, 3, stride=2, us=[True, False])
        self.GlobalAveragePooling2D = nn.AdaptiveAvgPool2d(1)

        self.dense1 = nn.Linear(256, 1024)
        self.dense2 = nn.Linear(1024, 512)
        self.dense3 = nn.Linear(512, 256)

        self.conv_func2 = conv_relu_in_block(128, self.nFilters*4, 1,padding=0, us=[True, False])
        self.conv_func3 = conv_relu_in_block(256, self.nFilters*2, 1,padding=0, us=[False, True])

        self.relu = nn.ReLU()

    def forward(self, x):
        # print('\n BEFORE zeropadding = ', x.shape)              #1, 128, 128, 128
        t = F.pad(x, (1,1,1,1))
        # print('global_blocktsize1 = ',t.shape)                  #1, 128, 130, 130
        t = self.conv_func1(t)
        # print('input of Global_AveragePooling 2D  = ', t.shape)  #1, 256, 65, 65
        t = self.GlobalAveragePooling2D(t)

        # print('after global Global_AveragePooling 2D \t', t.shape)    #1, 256, 1, 1
        t = t.squeeze(dim=2)
        # print('after global Global_AveragePooling 2D \t', t.shape)    #1, 256, 1
        t = t.squeeze(dim=2)
        # print('after global Global_AveragePooling 2D \t', t.shape)    #1, 256

        t = self.dense1(t)
        t =self.relu(t)
        # print('After dense1 ',t.shape)      #1, 1024
        t = self.dense2(t)
        t =self.relu(t)
        # print('After dense2 ',t.shape)      #1,512
        t = self.dense3(t)
        # print('After dense3 ',t.shape)      #1,256


        # print('\ninput of conv function = ',x.shape)        #1, 128,128,128
        _t = self.conv_func2(x)
        # print('input of mul1             = ',_t.shape, t.shape)   #1, 256, 128, 128
        t = t.unsqueeze(dim=2)
        t = t.unsqueeze(dim=2)
        # print('input of mul2             = ',_t.shape, t.shape)   #1, 256, 128, 128
        # print('-t[][][][] = ', _t[0][0][0:10][0:10])
        # print('t[][][][] = ',   t[0][0][0][0])


        # a = torch.randn(1,256,128,128)
        # b = torch.randn(1,256,1,1)
        # c = torch.mul(a,b)
        # print('a-t[][][][] = ', a[0][0][0:10][0:10])
        # print('bt[][][][] = ',  b[0][0][0][0])
        # print('c-t[][][][] = ', c[0][0][0:10][0:10])
        # print('abcabcabcabcshape    = ',a.shape, b.shape, c.shape)

        _t = torch.mul(_t,t)
        # print('\ninput of conv function second = ',_t.shape) # 256,128,128
        # print('-t[][][][] = ', _t[0][0][0:10][0:10])
        _t = self.conv_func3(_t)
        # print('after conv_func second = ', _t.shape) # 128,128,128  512,512,128
        return _t


class pos_block(nn.Module):
    def __init__(self,dilation):
        super().__init__()
        self.nFilters = 64
        self.dilation =dilation

        cat_shape=[128]
        self.conv_func1     = conv_relu_in_block(128, self.nFilters, 3, dilation = self.dilation[0],cat_shape=cat_shape)
        cat_shape2=[64]+cat_shape
        self.conv_func2     = conv_relu_in_block(192, self.nFilters, 3, dilation = self.dilation[1],cat_shape=cat_shape2)
        cat_shape3=[64]+cat_shape2
        self.conv_func3     = conv_relu_in_block(256, self.nFilters, 3, dilation = self.dilation[2],cat_shape=cat_shape3)
        cat_shape4=[64]+cat_shape3
        self.conv_func4     = conv_relu_in_block(320, self.nFilters, 3, dilation = self.dilation[3],cat_shape=cat_shape4)
        cat_shape5=[64]+cat_shape4
        self.conv_func5     = conv_relu_in_block(384, self.nFilters, 3, dilation = self.dilation[4],cat_shape=cat_shape5)
        cat_shape6=[64]+cat_shape5
        self.conv_func_last = conv_relu_in_block(448, self.nFilters*2, 1, padding=0,cat_shape=cat_shape6)

    def forward(self, x):
        t=x
        _t = self.conv_func1(t)
        t = torch.cat( [_t,t], dim=-3 ) # Default는 맨앞 여기서는 channel을 concat해야함. c,h,w
        _t = self.conv_func2(t)
        t = torch.cat( [_t,t], dim=-3 ) # Default는 맨앞 아마channel
        _t = self.conv_func3(t)
        t = torch.cat( [_t,t], dim=-3 ) # Default는 맨앞 아마channel
        _t = self.conv_func4(t)
        t = torch.cat( [_t,t], dim=-3 ) # Default는 맨앞 아마channel
        _t = self.conv_func5(t)
        t = torch.cat( [_t,t], dim=-3 ) # Default는 맨앞 아마channel

        t = self.conv_func_last(t)
        return t


