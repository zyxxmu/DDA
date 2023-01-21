"""A multi-thread tool to crop large images to sub-images for faster IO."""
"""https://github.com/XPixelGroup/ClassSR/blob/main/codes/data_scripts/extract_subimages_train.py"""
import os
import os.path as osp
import sys

from multiprocessing import Pool
import numpy as np
import cv2
from PIL import Image
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from utils import ProgressBar  # noqa: E402
import utils as data_util  # noqa: E402
import time
import torch
import pdb
import shutil
from Util.util_collections import im_score
save_list=["/userhome/dataset/fhdmi_class/train/class1/moire",
           "/userhome/dataset/fhdmi_class/train/class2/moire",
           "/userhome/dataset/fhdmi_class/train/class3/moire",
           "/userhome/dataset/fhdmi_class/train/class1/clear",
           "/userhome/dataset/fhdmi_class/train/class2/clear",
           "/userhome/dataset/fhdmi_class/train/class3/clear"]

clear_folder="/userhome/dataset/fhdmi_patches/train/clear/"
moire_folder="/userhome/dataset/fhdmi_patches/train/moire/"
score_list = {}
for i in save_list:
    if os.path.exists(i):
        pass
    else:
        os.makedirs(i)


M,N = [512,480]

H = np.zeros((M,N), dtype=np.float32)
D0 = 5
for u in range(M):
    for v in range(N):
        D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
        H[u,v] = np.exp(-D**2/(2*D0*D0))
HPF = 1 - H

def main():

    opt = {}
    opt['n_thread'] = 20
    # cut training data
    ########################################################################
    # check that all the clear and moire images have correct scale ratio
    img_moire_list = data_util._get_paths_from_images(moire_folder)
    img_clear_list = data_util._get_paths_from_images(clear_folder)
    
    print('processing...')
    opt['clear_folder'] = clear_folder
    opt['moire_folder'] = moire_folder
    extract_signle(opt)
       

def extract_signle(opt):
    input_folder = opt['moire_folder']

    img_list = data_util._get_paths_from_images(input_folder)
    #torch.multiprocessing.set_sharing_strategy('file_system')
        
    pbar = ProgressBar(len(img_list))
    #pool = Pool(20)

    for path in img_list:
        result = worker(path)
        score_list[result[0]] = result[1]
        pbar.update(result[2])

        #pool.apply_async(worker, args=(path,), callback=update)

    #pool.close()
    #pool.join()

    sort_result = sorted(score_list.items(),key=lambda x:(x[0][:-8],x[1]),reverse=False) # 先按照名字排序，再按照同名的排序score，排序结果是同名的patch都相连
    class_num = 3

    for i in range(0,len(sort_result),8):
        shutil.copy(osp.join(clear_folder, sort_result[i][0]), osp.join(save_list[5], sort_result[i][0]))
        shutil.copy(osp.join(moire_folder, sort_result[i][0]), osp.join(save_list[2], sort_result[i][0]))
        shutil.copy(osp.join(clear_folder, sort_result[i+1][0]), osp.join(save_list[5], sort_result[i+1][0]))
        shutil.copy(osp.join(moire_folder, sort_result[i+1][0]), osp.join(save_list[2], sort_result[i+1][0]))
        shutil.copy(osp.join(clear_folder, sort_result[i+2][0]), osp.join(save_list[5], sort_result[i+2][0]))
        shutil.copy(osp.join(moire_folder, sort_result[i+2][0]), osp.join(save_list[2], sort_result[i+2][0]))

    for i in range(3,len(sort_result),8):
        shutil.copy(osp.join(clear_folder, sort_result[i][0]), osp.join(save_list[4], sort_result[i][0]))
        shutil.copy(osp.join(moire_folder, sort_result[i][0]), osp.join(save_list[1], sort_result[i][0]))
        shutil.copy(osp.join(clear_folder, sort_result[i+1][0]), osp.join(save_list[4], sort_result[i+1][0]))
        shutil.copy(osp.join(moire_folder, sort_result[i+1][0]), osp.join(save_list[1], sort_result[i+1][0]))
        shutil.copy(osp.join(clear_folder, sort_result[i+2][0]), osp.join(save_list[4], sort_result[i+2][0]))
        shutil.copy(osp.join(moire_folder, sort_result[i+2][0]), osp.join(save_list[1], sort_result[i+2][0]))

    for i in range(5,len(sort_result),8):
        shutil.copy(osp.join(clear_folder, sort_result[i][0]), osp.join(save_list[3], sort_result[i][0]))
        shutil.copy(osp.join(moire_folder, sort_result[i][0]), osp.join(save_list[0], sort_result[i][0]))
        shutil.copy(osp.join(clear_folder, sort_result[i+1][0]), osp.join(save_list[3], sort_result[i+1][0]))
        shutil.copy(osp.join(moire_folder, sort_result[i+1][0]), osp.join(save_list[0], sort_result[i+1][0]))
        shutil.copy(osp.join(clear_folder, sort_result[i+2][0]), osp.join(save_list[3], sort_result[i+2][0]))
        shutil.copy(osp.join(moire_folder, sort_result[i+2][0]), osp.join(save_list[0], sort_result[i+2][0]))


def worker(path):
    img_name = osp.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    score = im_score(img, HPF)
    #print("Img {} Im score {}".format(path, score))
    return [img_name, score, 'Processing {:s} ...'.format(img_name)]



if __name__ == '__main__':
    main()