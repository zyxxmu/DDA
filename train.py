import argparse
import os
# import random,sys, matplotlib, torchvision
import time
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
# import matplotlib.pyplot as plt
from Util.util_collections import tensor2im, save_single_image, PSNR, Time2Str,setup_logging
from Net.LossNet import L1_LOSS, L1_Advanced_Sobel_Loss
from dataset.dataset import AIMMoire_dataset_test, AIMMoire_dataset, TIP2018moire_dataset_train, TIP2018moire_dataset_test,FHDMI_dataset,FHDMI_dataset_test
from torchnet import meter
from skimage.metrics import peak_signal_noise_ratio
from torchvision import transforms
import math
import logging
import random
import pdb
from test import val

def log(*args):
    args_list = map(str,args)
    tmp = ''.join(args_list)
    logging.info(tmp)

def train(args, model):
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    args.save_prefix = args.save_prefix+'/' +args.arch+'_'+args.dataset
    if not os.path.exists(args.save_prefix):    os.makedirs(args.save_prefix)
    setup_logging(os.path.join(args.save_prefix,'log.txt'))
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(args.save_prefix,'tflog'),comment=args.note)
        log(f'tensorboard path = \t\t\t{writer.log_dir}')
    log('torch devices = \t\t\t', args.device)
    log('save_path = \t\t\t\t', args.save_prefix)
    log(f'name: {args.name} note: {args.note}')

    args.pthfoler       = os.path.join( args.save_prefix , '1pth_folder/')
    args.psnrfolder     = os.path.join( args.save_prefix , '1psnr_folder/')
    if not os.path.exists(args.pthfoler)        :   os.makedirs(args.pthfoler)
    if not os.path.exists(args.psnrfolder)      :   os.makedirs(args.psnrfolder)

    if args.dataset == 'aim':
        train_dataset = AIMMoire_dataset
        test_dataset = AIMMoire_dataset_test   
    elif args.dataset == 'fhdmi':
        train_dataset = FHDMI_dataset
        test_dataset = FHDMI_dataset_test
    else:
        raise ValueError('no this dataset choise')

    Moiredata_train_class1 = train_dataset(args.traindata_path+"/class1",patch_size=args.patch_size)
    train_dataloader_class1 = DataLoader(Moiredata_train_class1,
                                  batch_size=args.batchsize,
                                  shuffle=True,
                                  num_workers=args.num_worker,
                                  drop_last=True)

    Moiredata_train_class2 = train_dataset(args.traindata_path+"/class2",patch_size=args.patch_size)
    train_dataloader_class2 = DataLoader(Moiredata_train_class2,
                                  batch_size=args.batchsize,
                                  shuffle=True,
                                  num_workers=args.num_worker,
                                  drop_last=True)
    
    Moiredata_train_class3 = train_dataset(args.traindata_path+"/class3",patch_size=args.patch_size)
    train_dataloader_class3 = DataLoader(Moiredata_train_class3,
                                  batch_size=args.batchsize,
                                  shuffle=True,
                                  num_workers=args.num_worker,
                                  drop_last=True)
    
    # split dataset into patches
    Moiredata_test = test_dataset(args.testdata_path+'/test')
    test_dataloader = DataLoader(Moiredata_test,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=args.num_worker,
                                 drop_last=False)

    lr = args.lr
    last_epoch = 0
    optimizer = optim.Adam(params=model.parameters(),
                           lr=lr )
    if args.arch == 'DMCNN':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=args.lr_step_decay)

    list_psnr_output = []
    list_loss_output = []

    model = nn.DataParallel(model)
    if len(args.resume)>0:
        log('load:')
        log(args.resume)
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt)

    if args.Train_pretrained_path:
        checkpoint = torch.load(args.Train_pretrained_path)
        model.load_state_dict(checkpoint['model'])
        last_epoch = checkpoint["epoch"]
        optimizer_state = checkpoint["optimizer"]
        optimizer.load_state_dict(optimizer_state)
        lr = checkpoint['lr']
        list_psnr_output = checkpoint['list_psnr_output']
        list_loss_output = checkpoint['list_loss_output']

    model = model.cuda()
    
    model.train()

    criterion_l1 = L1_LOSS()
    criterion_advanced_sobel_l1 = L1_Advanced_Sobel_Loss()

    psnr_meter  = meter.AverageValueMeter()
    Loss_meter1  = meter.AverageValueMeter()
    Loss_meter2 = meter.AverageValueMeter()
    Loss_meter3 = meter.AverageValueMeter()
    Loss_meter4 = meter.AverageValueMeter()

    for epoch in range(args.max_epoch):
        train_loader_1 = iter(train_dataloader_class1)
        train_loader_2 = iter(train_dataloader_class2)
        train_loader_3 = iter(train_dataloader_class3)

        batch_num = len(train_dataloader_class1) * 3

        if epoch < last_epoch:
            continue
        log('\nepoch = {} / {}'.format(epoch + 1, args.max_epoch))
        start = time.time()
                
        Loss_meter1.reset()
        Loss_meter2.reset()
        Loss_meter3.reset()
        Loss_meter4.reset()
        psnr_meter.reset()
        for ii in range(0, batch_num):
            if ii % 3 == 0:
                model.apply(lambda m: setattr(m, 'width_mult', args.width_list[0]))
                moires, clears_list, labels = next(train_loader_1)
            elif ii % 3 == 1:
                model.apply(lambda m: setattr(m, 'width_mult', args.width_list[1]))
                moires, clears_list, labels = next(train_loader_2)
            elif ii % 3 == 2:
                model.apply(lambda m: setattr(m, 'width_mult', args.width_list[2]))
                moires, clears_list, labels = next(train_loader_3)

            moires = moires.cuda()
            clear3, clear2, clear1 = clears_list

            clear3 = clear3.to(args.device)
            clear2 = clear2.to(args.device)
            clear1 = clear1.to(args.device)
            if args.arch == 'MBCNN':
                
                # clear1 = clear1.cuda()

                output3, output2, output1 = model(moires) # 32,1,256,256 = 32,1,256,256
                # output1 = model(moires)

                Loss_l1                  = criterion_l1(output1, clear1)
                Loss_advanced_sobel_l1   = criterion_advanced_sobel_l1(output1, clear1)
                Loss_l12                 = criterion_l1(output2, clear2)
                Loss_advanced_sobel_l12  = criterion_advanced_sobel_l1(output2, clear2)
                Loss_l13                 = criterion_l1(output3, clear3)
                Loss_advanced_sobel_l13  = criterion_advanced_sobel_l1(output3, clear3)

                Loss1 = Loss_l1  + (0.25)*Loss_advanced_sobel_l1
                Loss2 = Loss_l12 + (0.25)*Loss_advanced_sobel_l12
                Loss3 = Loss_l13 + (0.25)*Loss_advanced_sobel_l13

                loss = Loss1 + Loss2 + Loss3

                loss_check1 = Loss1
                loss_check2 = Loss_l1
                loss_check3 = (0.25)*Loss_advanced_sobel_l1
                optimizer.zero_grad()
                loss.backward()            # loss.backward(retain_graph = True) # retain_graph = True
                optimizer.step()

                moires = tensor2im(moires)
                output1 = tensor2im(output1)
                clear1 = tensor2im(clear1)

                psnr = peak_signal_noise_ratio(output1,clear1)
                psnr_meter.add(psnr)
                Loss_meter1.add(loss.item())
                Loss_meter2.add(loss_check1.item())
                Loss_meter3.add(loss_check2.item())
                Loss_meter4.add(loss_check3.item())
            else:
                # clear1 = clear1.cuda()

                output1 = model(moires) # 32,1,256,256 = 32,1,256,256
                # output1 = model(moires)

                Loss_l1 = criterion_l1(output1, clear1)

                Loss1 = Loss_l1

                loss = Loss1

                loss_check1 = Loss1
                loss_check2 = Loss_l1
                optimizer.zero_grad()
                loss.backward()        
                optimizer.step()

                moires = tensor2im(moires)

                output1 = tensor2im(output1)
                clear1 = tensor2im(clear1)

                psnr = peak_signal_noise_ratio(output1,clear1)
                psnr_meter.add(psnr)
                Loss_meter1.add(loss.item())
                Loss_meter2.add(loss_check1.item())
                Loss_meter3.add(loss_check2.item())
                # break

            if ii%1 == 0:
                log('iter: {} \ttraining set : \tPSNR = {:f}\t loss = {:f}\t Loss1(scale) = {:f} \t Loss_L1 = {:f} + Loss_sobel = {:f},\t '
                    .format(ii, psnr_meter.value()[0], Loss_meter1.value()[0],  Loss_meter2.value()[0], Loss_meter3.value()[0], Loss_meter4.value()[0] ))
            # if ii>2:break

        psnr_output = val(model, test_dataloader, args)

        if args.tensorboard:
            tf_dict = dict(
                training_psnr=psnr_meter.value()[0],
                training_loss=Loss_meter1.value()[0],
                training_Loss1=Loss_meter2.value()[0],
                training_Loss_L1=Loss_meter3.value()[0],
                training_Loss_sobel=Loss_meter4.value()[0],
                val_psnr=psnr_output,

            )
        list_psnr_output.append(round(psnr_output,5) )
        if args.arch == 'DMCNN':
            scheduler.step()
        else:
            if epoch > 5:
                list_tmp = list_psnr_output[-5:]
                for j in range(4):
                    sub = 10 * (math.log10( ( list_tmp[j] / list_tmp[j+1] ) ))
                    if sub > 0.001: break
                    if j == 3:
                        log('\033[30m \033[41m' + 'LR was Decreased!!!{:} > {:}\t\t\t\t\t\t\t\t\t'.format(lr,lr/2) + '\033[0m' )
                        lr = lr * 0.5
                        for param_group in optimizer.param_groups:  param_group['lr'] = lr
                    if lr < 1e-6:                                   exit()

        if psnr_output > args.bestperformance: 
            args.bestperformance = psnr_output
            file_name = args.pthfoler + 'ckpt_best.pth'
            torch.save(model.state_dict(), file_name)
            log('\033[30m \033[42m' + 'PSNR WAS UPDATED! '+'\033[0m')

        if (epoch + 1) % args.save_every == 0 or epoch == 0:
            file_name = args.pthfoler + 'ckpt_last.pth'
            checkpoint = {  'epoch': epoch + 1,
                            "optimizer": optimizer.state_dict(),
                            "model": model.state_dict(),
                            "lr": lr,
                            "list_psnr_output": list_psnr_output,
                            "list_loss_output": list_loss_output,
                            }
            torch.save(checkpoint, file_name)

            with open(args.save_prefix + "/1_PSNR_validation_set_output_psnr.txt", 'w') as f:
                f.write("psnr_output: {:}\n".format(list_psnr_output))
            with open(args.save_prefix + "/1_Loss_validation_set_output_loss.txt", 'w') as f:
                f.write("loss_output: {:}\n".format(list_loss_output))

        if epoch == (args.max_epoch-1):
            file_name2 = args.pthfoler + '{0}_stdc_epoch{1}.pth'.format(args.name, epoch + 1)
            torch.save(model.state_dict(), file_name2)


        log('1 epoch spends:{:.2f}sec\t remain {:2d}:{:2d} hours'.format(
            (time.time() - start),
            int((args.max_epoch - epoch) * (time.time() - start) // 3600) ,
            int((args.max_epoch - epoch) * (time.time() - start) % 3600 / 60 ) ))

    return "Training Finished!"


