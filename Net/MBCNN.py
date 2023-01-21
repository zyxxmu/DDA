import torch
import torch.nn as nn
import torch.nn.functional as F
from Net.MBCNN_class import *
import torch.nn.functional as F
import pdb

class MBCNN(nn.Module):
    def __init__(self, nFilters, multi=True):
        super().__init__()
        self.imagesize = 256
        self.sigmoid = nn.Sigmoid()
        self.Space2Depth1 = nn.PixelUnshuffle(2)
        self.Depth2space1 = nn.PixelShuffle(2)

        self.conv_func1 = conv_relu1(12, nFilters * 2, 3, padding=1)
        self.pre_block1 = pre_block((1, 2, 3, 2, 1))
        self.conv_func2 = conv_relu1(128, nFilters * 2, 3, padding=0, stride=2)
        self.pre_block2 = pre_block((1, 2, 3, 2, 1))

        self.conv_func3 = conv_relu1(128, nFilters * 2, 3, padding=0, stride=2)
        self.pre_block3 = pre_block((1, 2, 2, 2, 1))
        self.global_block1 = global_block(self.imagesize // 8)
        self.pos_block1 = pos_block((1, 2, 2, 2, 1))
        self.conv1 = conv1(128, 12, 3,us=[True,False])

        self.conv_func4 = conv_relu1(131, nFilters * 2, 1, padding=0,cat_shape=(3,nFilters*2),set_cat_mul=(False,True))
        self.global_block2 = global_block(self.imagesize // 4)
        self.pre_block4 = pre_block((1, 2, 3, 2, 1))
        self.global_block3 = global_block(self.imagesize // 4)
        self.pos_block2 = pos_block((1, 2, 3, 2, 1))
        self.conv2 = conv1(128, 12, 3,us=[True,False])

        self.conv_func5 = conv_relu1(131, nFilters * 2, 1, padding=0,cat_shape=(3,nFilters*2),set_cat_mul=(False,True))

        self.global_block4 = global_block(self.imagesize // 2)
        self.pre_block5 = pre_block((1, 2, 3, 2, 1))
        self.global_block5 = global_block(self.imagesize // 2)
        self.pos_block3 = pos_block((1, 2, 3, 2, 1))
        self.conv3 = conv1(128, 12, 3,us=[True,False])

    def forward(self, x):
        output_list = []
        shape = list(x.shape)
        # pdb.set_trace()
        # print('input shape = ',shape)
        batch, channel, height, width = shape
        # print('\nbatch , channel, height, width \t',batch, channel, height, width) # 2, 3, 256, 256

        # print('line5858585858.shape\t',x.shape) #  2,12,128,128
        _x = self.Space2Depth1(x)

        # print('line78line78line78t1.shape\t',_x.shape) #  2,12,128,128
        t1 = self.conv_func1(_x)
        # print('line80line80line80t1.shape',t1.shape)
        t1 = self.pre_block1(t1)
        # print('line83   line83  line83  t1.shape',t1.shape)
        t2 = F.pad(t1, (1, 1, 1, 1))

        # print('\nFlag1 convfunc22 line 86, line 86.line 86',t2.shape)  #4,128,514,514
        t2 = self.conv_func2(t2)
        # print('line 88, line 88.line 88',t2.shape)  #4,128,256,256,
        t2 = self.pre_block2(t2)
        # print('line 90, line 90.line 90',t2.shape)#4,128,256,256,
        t3 = F.pad(t2, (1, 1, 1, 1))

        # print('\nFlag2 convfunc33 line 91, line 91.line 91',t3.shape) # 128,258,258
        t3 = self.conv_func3(t3)
        # print('line 93, line 93.line 93',t3.shape)      # 128,128,128 check
        t3 = self.pre_block3(t3)
        # print('line 95,global block input',t3.shape)      #128,128,128

        t3 = self.global_block1(t3)
        # print('line 97, line 97.line 97',t3.shape)
        t3 = self.pos_block1(t3)
        # print('line t3, line t3.line t3',t3.shape)
        t3_out = self.conv1(t3)
        # print('line t3_out, line t3_out.line t3_out', t3_out.shape)
        t3_out = self.Depth2space1(t3_out)
        # print('t3_out.shape '         , t3_out.shape)
        t3_out = F.sigmoid(t3_out)
        output_list.append(t3_out)

        _t2 = torch.cat([t3_out, t2], dim=-3)  # channel을 concat하기
        # print('\noutput torch.cat ',_t2.shape)  #
        _t2 = self.conv_func4(_t2)
        # print('output conv_func4 ',_t2.shape) # 4,131,512,512
        _t2 = self.global_block2(_t2)
        # print('output global_block2 ',_t2.shape) # 4,131,512,512
        _t2 = self.pre_block4(_t2)
        # print('output pre_block4 ',_t2.shape) # 4,131,512,512
        _t2 = self.global_block3(_t2)
        # print('output global_block3 ',_t2.shape) # 4,131,512,512
        _t2 = self.pos_block2(_t2)
        # print('output pos_block2 ',_t2.shape) # 4,131,512,512
        t2_out = self.conv2(_t2)
        # print('output conv2 ',t2_out.shape) # 4,131,512,512
        t2_out = self.Depth2space1(t2_out)
        # print('output Depth2space2 ',t2_out.shape) # 4,131,512,512
        t2_out = F.sigmoid(t2_out)
        output_list.append(t2_out)
        # print('\n\nline 118 ')

        _t1 = torch.cat([t1, t2_out], dim=-3)  # channel을 concat하기
        # print('\noutput torch.cat ',_t1.shape) # 4,131,512,512
        _t1 = self.conv_func5(_t1)
        # print('output conv_func5  ',_t1.shape)    #4,128,512,512
        _t1 = self.global_block4(_t1)
        # print('output global_block4 ',_t1.shape)
        _t1 = self.pre_block5(_t1)
        # print('output pre_block5  ',_t1.shape)
        _t1 = self.global_block5(_t1)
        # print('output glob/al_block5 ',_t1.shape)
        _t1 = self.pos_block3(_t1)
        # print('output global_block5 ',_t1.shape)
        _t1 = self.conv3(_t1)
        # print('output conv3 ',_t1.shape)
        y = self.Depth2space1(_t1)
        # print('output Depth2space1 ',y.shape)

        y = self.sigmoid(y)
        output_list.append(y)
        # print('MBCNN Train finished')

        return output_list
