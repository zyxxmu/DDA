#main.py
import argparse
import logging
import os
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from train import train
from Net.MCNN import MoireCNN
from Net.MBCNN import MBCNN
from test import test

parser = argparse.ArgumentParser()

#*********** dataset
parser.add_argument('--traindata_path', type=str, #train10000
                    default= None)
parser.add_argument('--testdata_path', type=str,
                    default= None)
parser.add_argument('--testmode_path', type=str,
                    default= None)
parser.add_argument('--dataset',default='fhdmi',help='set dataset')
#********** training detail
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--num_worker', type=int, default=8,
                    help='number of workers')
parser.add_argument('--batchsize', type=int,default= 32,
                    help='mini batch size')
parser.add_argument('--max_epoch', type=int, default=50,
                    help='number of max_epoch')
parser.add_argument('--save_every', type=int,default= 10,
                    help='saving period for pretrained weight ')
parser.add_argument('--val_resize', type=int, nargs='+',default=tuple(),
                    help='val resize shape')

#********** experiment set
parser.add_argument('--name', type=str,default='MBCNN',
                    help='name for this experiment rate')
parser.add_argument('--arch', type=str,default='MBCNN',
                    help='network arch, default:MBCNN, optional:DMCNN')
parser.add_argument('--note', default='', help='set a note for this experiment')
parser.add_argument('--device', type=str, default='cuda or cpu',
                    help='device, define it first!!')
parser.add_argument('--save_prefix', type=str, default='./result',
                    help='saving folder directory')
parser.add_argument('--bestperformance', type=float, default=0.,
                    help='saving folder directory')
parser.add_argument('--Train_pretrained_path', type=str, default = None,
                    help='saving folder directory')
parser.add_argument('--Test_pretrained_path', type=str, default = None,
                    help='saving folder directory')
parser.add_argument('--resume',default='',help='resume [pth]')
parser.add_argument('--patch_size',default=128,type=int,help='set patch size')
parser.add_argument('--seed',default=1111,type=int,help='random seed')
parser.add_argument('--test',action='store_true',default='test or not')
parser.add_argument('--shffule_class',action='store_true',help='shuffle class or not')
parser.add_argument('--shuffle_class',action='store_true',help='shuffle class train')
parser.add_argument('--operation',default='train',help='train|test|merge_pre|merge')
#********** meric
parser.add_argument('--psnr_axis_min', type=int,default=10,
                    help='mininum line for psnr graph')
parser.add_argument('--psnr_axis_max', type=int,default=70,
                    help='maximum line for psnr graph')
parser.add_argument('--psnrfolder', type=str,default='psnrfoler path was not configured',
                    help='psnrfoler path, define it first!!')

parser.add_argument(
    '--lr_step_decay',
    type=int,
    nargs='+',
    default=[50, 100],
    help='the iterval of learn rate. default:50, 100'
)

parser.add_argument(
    '--width_list',
    type=float,
    nargs='+',
    default=[0.25, 0.5, 0.75],
    help='width_list'
)
#********** tensorboard
parser.add_argument('--tensorboard',action='store_true',help='set tensorboard record')

def set_seed(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args = parser.parse_args()
if __name__ == "__main__":
    set_seed(args.seed)
    if args.arch == 'MBCNN':
        net = MBCNN(64)
    else:
        net = MoireCNN()
    
    if args.operation == 'train':
        train(args, net)
    elif args.operation == 'test':
        test(args, net)

