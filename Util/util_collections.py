import os
import numpy as np
import torch
import torch.nn as nn
import colour
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from torch.autograd import Variable
import math
import time
import logging

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
def setup_logging(log_file='log.txt',filemode='w'):
    """Setup logging configuration
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode=filemode)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def crop_cpu(img, crop_sz, step):
    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    lr_list=[]
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            lr_list.append(crop_img)
    h=x + crop_sz
    w=y + crop_sz
    return lr_list, num_h, num_w, h, w

def combine(sr_list, num_h, num_w, h, w, patch_size, step):
    index=0
    sr_img = np.zeros((h*self.scale, w*self.scale, 3), 'float32')
    for i in range(num_h):
        for j in range(num_w):
            sr_img[i*step*self.scale:i*step*self.scale+patch_size*self.scale,j*step*self.scale:j*step*self.scale+patch_size*self.scale,:]+=sr_list[index]
            index+=1
    sr_img=sr_img.astype('float32')

    for j in range(1,num_w):
        sr_img[:,j*step*self.scale:j*step*self.scale+(patch_size-step)*self.scale,:]/=2

    for i in range(1,num_h):
        sr_img[i*step*self.scale:i*step*self.scale+(patch_size-step)*self.scale,:,:]/=2
    return sr_img

def combine_addmask(self, sr_list, num_h, num_w, h, w, patch_size, step, _type):
    index = 0
    sr_img = np.zeros((h * self.scale, w * self.scale, 3), 'float32')

    for i in range(num_h):
        for j in range(num_w):
            sr_img[i * step * self.scale:i * step * self.scale + patch_size * self.scale,
            j * step * self.scale:j * step * self.scale + patch_size * self.scale, :] += sr_list[index]
            index += 1
    sr_img = sr_img.astype('float32')

    for j in range(1, num_w):
        sr_img[:, j * step * self.scale:j * step * self.scale + (patch_size - step) * self.scale, :] /= 2

    for i in range(1, num_h):
        sr_img[i * step * self.scale:i * step * self.scale + (patch_size - step) * self.scale, :, :] /= 2

    index2 = 0
    for i in range(num_h):
        for j in range(num_w):
            # add_mask
            alpha = 1
            beta = 0.2
            gamma = 0
            bbox1 = [j * step * self.scale + 8, i * step * self.scale + 8,
                        j * step * self.scale + patch_size * self.scale - 9,
                        i * step * self.scale + patch_size * self.scale - 9]  # xl,yl,xr,yr
            zeros1 = np.zeros((sr_img.shape), 'float32')

            if torch.max(_type, 1)[1].data.squeeze()[index2] == 0:
                mask2 = cv2.rectangle(zeros1, (bbox1[0]+1, bbox1[1]+1), (bbox1[2]-1, bbox1[3]-1), color=(0, 255, 0), thickness=-1)# simple green
            elif torch.max(_type, 1)[1].data.squeeze()[index2] == 1:
                mask2 = cv2.rectangle(zeros1, (bbox1[0]+1, bbox1[1]+1), (bbox1[2]-1, bbox1[3]-1), color=(0, 255, 255), thickness=-1)# medium yellow
            elif torch.max(_type, 1)[1].data.squeeze()[index2] == 2:
                mask2 = cv2.rectangle(zeros1, (bbox1[0]+1, bbox1[1]+1), (bbox1[2]-1, bbox1[3]-1), color=(0, 0, 255), thickness=-1)# hard red

            sr_img = cv2.addWeighted(sr_img, alpha, mask2, beta, gamma)
            index2+=1
    return sr_img

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in [50, 100, 150,200]:
        state['lr'] *= 0.3
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


def Time2Str():
    sec = time.time()
    tm = time.localtime(sec)
    time_str = '21'+'{:02d}'.format(tm.tm_mon)+'{:02d}'.format(tm.tm_mday) +'_'+'{:02d}'.format(tm.tm_hour+9)+':{:02d}'.format(tm.tm_min)
    return time_str


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in [50, 100, 150,200]:
        state['lr'] *= 0.3
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


def tensor2im(input_image, imtype=np.uint8):

    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.detach() # true
    else:
        return input_image

    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.swapaxes(image_numpy, -3,-2)
    image_numpy = np.swapaxes(image_numpy, -2,-1)

    image_numpy = image_numpy * 255

    #simage_numpy = (image_numpy + 1.0) / 2.0

    return image_numpy.astype(imtype)


def PSNR(original, contrast): # metrics.peak_signal_noise_ratio랑 동일

    original = original*255.
    contrast = contrast*255.

    mse = np.mean((original - contrast) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR


def save_single_image(img, img_path):
    # img = np.transpose(img, (1, 2, 0))

    if np.shape(img)[-1] ==1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # img = cv2.resize(img, dsize=( 1170,2532 ), interpolation=cv2.INTER_NEAREST )
    # img = cv2.resize(img, dsize=( 1080,2340 ), interpolation=cv2.INTER_NEAREST )
    img = img * 255

    cv2.imwrite(img_path, img)
    # return img


def pixel_unshuffle(batch_input, shuffle_scale = 2, device=torch.device('cuda')):
    batch_size = batch_input.shape[0]
    num_channels = batch_input.shape[1]
    height = batch_input.shape[2]
    width = batch_input.shape[3]

    conv1 = nn.Conv2d(1, 1, 2, 2, bias=False)
    conv1 = conv1.to(device)
    conv1.weight.data = torch.from_numpy(np.array([[1, 0],
                                                    [0, 0]], dtype='float32').reshape((1, 1, 2, 2))).to(device)

    conv2 = nn.Conv2d(1, 1, 2, 2, bias=False)
    conv2 = conv2.to(device)
    conv2.weight.data = torch.from_numpy(np.array([[0, 1],
                                                    [0, 0]], dtype='float32').reshape((1, 1, 2, 2))).to(device)
    conv3 = nn.Conv2d(1, 1, 2, 2, bias=False)
    conv3 = conv3.to(device)
    conv3.weight.data = torch.from_numpy(np.array([[0, 0],
                                                    [1, 0]], dtype='float32').reshape((1, 1, 2, 2))).to(device)
    conv4 = nn.Conv2d(1, 1, 2, 2, bias=False)
    conv4 = conv4.to(device)
    conv4.weight.data = torch.from_numpy(np.array([[0, 0],
                                                    [0, 1]], dtype='float32').reshape((1, 1, 2, 2))).to(device)
    Unshuffle = torch.ones((batch_size, 4, height//2, width//2), requires_grad=False).to(device)

    for i in range(num_channels):
        each_channel = batch_input[:, i:i+1, :, :]
        first_channel = conv1(each_channel)
        second_channel = conv2(each_channel)
        third_channel = conv3(each_channel)
        fourth_channel = conv4(each_channel)
        result = torch.cat((first_channel, second_channel, third_channel, fourth_channel), dim=1)
        Unshuffle = torch.cat((Unshuffle, result), dim=1)

    Unshuffle = Unshuffle[:, 4:, :, :]
    return Unshuffle.detach()


def default_loader(path):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    region = img.crop((1+int(0.15*w), 1+int(0.15*h), int(0.85*w), int(0.85*h)))
    return region


def calc_pasnr_from_folder(src_path, dst_path):
    src_image_name = os.listdir(src_path)
    dst_image_name = os.listdir(dst_path)
    image_label = ['_'.join(i.split("_")[:-1]) for i in src_image_name]
    num_image = len(src_image_name)
    psnr = 0
    for ii, label in tqdm(enumerate(image_label)):
        src = os.path.join(src_path, "{}_source.png".format(label))
        dst = os.path.join(dst_path, "{}_target.png".format(label))
        src_image = default_loader(src)
        dst_image = default_loader(dst)

        single_psnr = colour.utilities.metric_psnr(src_image, dst_image, 255)
        psnr += single_psnr

    psnr /= num_image
    return psnr


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[0] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[i], img2[i]))
            return np.array(ssims).mean()
        elif img1.shape[0] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


reconstruction_function = nn.MSELoss(size_average=False)
def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD

def get_color(img):
	img = cv2.resize(img,(256,256))
	B = img[:,:,0]
	G = img[:,:,1]
	R = img[:,:,2]
	rg = torch.Tensor(R - G)
	yb = torch.Tensor(0.5*(R+G) - B)

	dev_rg = torch.std(rg)
	dev_yb = torch.std(yb)
	mean_rg = torch.mean(rg)
	mean_yb = torch.mean(yb)
	dev = torch.sqrt(dev_rg*dev_rg + dev_yb*dev_yb)
	mean = torch.sqrt(mean_rg*mean_rg + mean_yb*mean_yb)
	M = dev + 0.3*mean
	return M

def Gaussian_score(image, HPF):
    f = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # transform the image into frequency domain, f --> F
    F = np.fft.fft2(f)
    Fshift = np.fft.fftshift(F)

    # Image Filters
    Gshift = Fshift * HPF
    G = np.fft.ifftshift(Gshift)
    g = np.abs(np.fft.ifft2(G))
    g = cv2.convertScaleAbs(g)
    return torch.mean(torch.Tensor(g))

def im_score(image, HPF):
	color = get_color(image)
	frequency = Gaussian_score(image, HPF)
	im_score = color*frequency
	return im_score