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

#Crop AIM19 into 256x256 patchsize

def main():
    mode = 'pair'  # single (one input folder) | pair (extract corresponding Moire and Clean pairs)
    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 3  # 3 is the default value in cv2
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.
    clear_folder = '/userhome/dataset/fhdmi/train/target'# fix to your path
    moire_folder = '/userhome/dataset/fhdmi/train/source'# fix to your path
    save_clear_folder = '/userhome/dataset/fhdmi_patches/train/clear/'
    save_moire_folder = '/userhome/dataset/fhdmi_patches/train/moire'

    crop_sz = (480,512)  # the size of each sub-image (clear)
    step = (480,512)  # step of the sliding crop window (clear)
    thres_sz = 96  # size threshold
    ########################################################################
    # check that all the clear and moire images have correct scale ratio
    img_moire_list = data_util._get_paths_from_images(moire_folder)
    img_clear_list = data_util._get_paths_from_images(clear_folder)
    assert len(img_clear_list) == len(img_moire_list), 'different lenclearh of clear_folder and moire_folder.'
    #for path_clear, path_moire in zip(img_clear_list, img_moire_list):
    #    image_clear = Image.open(path_clear)
    #    image_moire = Image.open(path_moire)
    #    w_clear, h_clear =  image_clear.size
    #    w_moire, h_moire = image_moire.size
    #    assert w_clear / w_moire == 1, 'clear width [{:d}] is not the same as moire width [{:d}] for {:s}.'.format(  # noqa: E501
    #        w_clear, w_moire, path_clear)
    #    assert h_clear / h_moire ==1, 'clear height [{:d}] is not the same as moire height [{:d}] for {:s}.'.format(  # noqa: E501
    #        h_clear, h_moire, path_clear)
    # check crop size, step and threshold size
    print('process clear...')
    opt['input_folder'] = clear_folder
    opt['save_folder'] = save_clear_folder
    opt['crop_sz'] = crop_sz
    opt['step'] = step
    opt['thres_sz'] = thres_sz
    extract_signle(opt)
    print('process moire...')
    opt['input_folder'] = moire_folder
    opt['save_folder'] = save_moire_folder
    opt['crop_sz'] = crop_sz
    opt['step'] = step
    opt['thres_sz'] = thres_sz 
    extract_signle(opt)
    assert len(data_util._get_paths_from_images(save_clear_folder)) == len(
        data_util._get_paths_from_images(
            save_moire_folder)), 'different lenclearh of save_clear_folder and save_moire_folder.'



def extract_signle(opt):
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))
    else:
        print('Folder [{:s}] amoireeady exists. Exit...'.format(save_folder))
        sys.exit(1)
    img_list = data_util._get_paths_from_images(input_folder)

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(len(img_list))

    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')


def worker(path, opt):
    crop_sz = opt['crop_sz']
    if isinstance(crop_sz,int):
        crop_sz_w = crop_sz
        crop_sz_h = crop_sz
    else:
        crop_sz_w,crop_sz_h = crop_sz
    step = opt['step']
    if isinstance(crop_sz,int):
        step_w = step
        step_h = step
    else:
        step_w,step_h = step
    thres_sz = opt['thres_sz']
    img_name = osp.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img,[1920,1024])
    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    h_space = np.arange(0, h - crop_sz_h + 1, step_h)
    if h - (h_space[-1] + crop_sz_h) > thres_sz:
        h_space = np.append(h_space, h - crop_sz_h)
    w_space = np.arange(0, w - crop_sz_w + 1, step_w)
    if w - (w_space[-1] + crop_sz_w) > thres_sz:
        w_space = np.append(w_space, w - crop_sz_w)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz_h, y:y + crop_sz_w]
            else:
                crop_img = img[x:x + crop_sz_h, y:y + crop_sz_w, :]
            crop_img = np.ascontiguousarray(crop_img)
            cv2.imwrite(
                osp.join(opt['save_folder'],
                         img_name.replace('src_','').replace('tar_','').replace('.png', '_s{:03d}.png'.format(index))), crop_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    main()