import os
import random

# import h5py
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from pathlib import Path
import pdb
def default_loader(path):
    img = Image.open(path).convert('RGB')
    # img = Image.open(path).convert('L')
    w, h = img.size

    # region = img.crop((1+int(0.15*w), 1+int(0.15*h), int(0.85*w), int(0.85*h)))
    # region = img.crop( (156,156,660,660) )
    return img

def default_loader_crop(path):
    img = Image.open(path).convert('RGB')
    # img = Image.open(path).convert('L')

    # region = img.crop((1+int(0.15*w), 1+int(0.15*h), int(0.85*w), int(0.85*h)))
    region = img.crop( (156,156,660,660) )
    return region

class Moire_dataset(Dataset):
    def __init__(self, root, loader = default_loader):
        moire_data_root = os.path.join(root, 'source')
        clear_data_root = os.path.join(root, 'target')

        image_names = os.listdir(clear_data_root)
        image_names = ["_".join(i.split("_")[:-1]) for i in image_names]
        # image_names = "_".join(image_names.split("_")[:-1])

        self.moire_images = [os.path.join(moire_data_root, x + '_source.png') for x in image_names]
        self.clear_images = [os.path.join(clear_data_root, x + '_target.png') for x in image_names]
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.loader = loader
        self.labels = image_names

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        clear_img_path = self.clear_images[index]
        moire = self.loader(moire_img_path)
        clear = self.loader(clear_img_path)
        moire = self.transforms(moire)
        clear = self.transforms(clear)
        label = self.labels[index]
        return moire, clear, label

    def __len__(self):
        return len(self.moire_images)

#
class AIMMoire_dataset(Dataset):
    def __init__(self, root, crop = True, loader = default_loader,patch_size=256):
        self.crop = crop
        moire_data_root = os.path.join(root, 'moire')
        clear_data_root = os.path.join(root, 'clear')
        self.patch_size = (patch_size,patch_size)
        image_names1 = os.listdir(moire_data_root)
        image_names1.sort()
        self.moire_images = [os.path.join(moire_data_root, x ) for x in image_names1]

        image_names2 = os.listdir(clear_data_root)
        image_names2.sort()
        self.clear_images = [os.path.join(clear_data_root, x ) for x in image_names2]

        image_names1 = [".".join(i.split(".")[:-1]) for i in image_names1]

        self.transforms = transforms.Compose([
            # transforms.Resize((400, 200)),
            transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transforms2 = transforms.Compose([
            transforms.Resize((patch_size//2, patch_size//2)),
        ])
        self.transforms3 = transforms.Compose([
            transforms.Resize((patch_size//4, patch_size//4)),
        ])
        # self.transforms3 = transforms.CenterCrop(128,128)


        self.loader = loader
        self.labels = image_names1

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        clear_img_path = self.clear_images[index]
        moire = self.loader(moire_img_path)
        clear = self.loader(clear_img_path)
        moire = self.transforms(moire)
        clear = self.transforms(clear)


        if self.crop == True:
            i, j, h, w = transforms.RandomCrop.get_params(moire, output_size=self.patch_size )
            moire = TF.crop(moire, i, j, h, w)
            clear = TF.crop(clear, i, j, h, w)

        clear2 = self.transforms2(clear)
        clear3 = self.transforms3(clear2)
        clear_list = [clear3, clear2, clear]
        label = self.labels[index]
        return moire, clear_list, label

    def __len__(self):
        return len(self.moire_images)

class Enhancer_Moire_dataset(Dataset):
    def __init__(self, moire_root,ori_root, clear_root, loader = default_loader):
        moire_data_root = os.path.join(moire_root)
        clear_data_root = os.path.join(clear_root)
        orig_data_root = os.path.join(ori_root)

        self.moire_images = sorted(os.listdir(moire_data_root))
        self.clear_images = sorted(os.listdir(clear_data_root))
        self.ori_images = sorted(os.listdir(orig_data_root))

        image_names = ["_".join(i.split("_")[:-1]) for i in self.clear_images]
        # image_names = "_".join(image_names.split("_")[:-1])

        self.moire_images = [os.path.join(moire_data_root, x) for x in self.moire_images]
        self.clear_images = [os.path.join(clear_data_root, x) for x in self.clear_images]
        self.ori_images = [os.path.join(orig_data_root, x) for x in self.ori_images]
        self.transforms = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.loader = loader
        self.labels = image_names

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        clear_img_path = self.clear_images[index]
        ori_img_path = self.ori_images[index]
        moire = self.loader(moire_img_path)
        clear = self.loader(clear_img_path)
        ori = self.loader(ori_img_path)
        moire = self.transforms(moire)
        ori = self.transforms(ori)
        clear = self.transforms(clear)
        label = self.labels[index]
        return moire,ori,clear, label

    def __len__(self):
        return len(self.moire_images)


## image testteststest
class AIMMoire_dataset_test(Dataset):
    def __init__(self, root, crop = False, loader = default_loader):
        self.crop = crop
        moire_data_root = os.path.join(root, 'moire')
        clear_data_root = os.path.join(root, 'clear')

        image_names1 = os.listdir(moire_data_root)
        image_names1.sort()
        self.moire_images = [os.path.join(moire_data_root, x ) for x in image_names1]

        image_names2 = os.listdir(clear_data_root)
        image_names2.sort()
        self.clear_images = [os.path.join(clear_data_root, x ) for x in image_names2]

        image_names1 = [".".join(i.split(".")[:-1]) for i in image_names1]

        self.transforms = transforms.Compose([
            # transforms.Resize((400, 200)),
            transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transforms2 = transforms.Compose([
            transforms.Resize((64, 64)),
        ])
        self.transforms3 = transforms.Compose([
            transforms.Resize((32, 32)),
        ])

        self.loader = loader
        self.labels = image_names1

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        clear_img_path = self.clear_images[index]
        moire = self.loader(moire_img_path)
        clear = self.loader(clear_img_path)
        moire = self.transforms(moire)
        clear = self.transforms(clear)


        # clear2 = self.transforms2(clear)
        # clear3 = self.transforms3(clear2)

        # clear_list = [clear3, clear2, clear]
        clear_list = [torch.tensor(0), torch.tensor(0), clear]
        label = self.labels[index]
        return moire, clear_list, label

    def __len__(self):
        return len(self.moire_images)



class FHDMI_dataset(Dataset):
    def __init__(self, root, crop = True, loader = default_loader,resize=None, patch_size=384):
        self.crop = crop
        moire_data_root = os.path.join(root, 'moire')
        clear_data_root = os.path.join(root, 'clear')
        self.patch_size = patch_size
        image_names1 = os.listdir(moire_data_root)
        image_names1.sort()
        self.moire_images = [os.path.join(moire_data_root, x ) for x in image_names1]

        image_names2 = os.listdir(clear_data_root)
        image_names2.sort()
        self.clear_images = [os.path.join(clear_data_root, x ) for x in image_names2]

        image_names1 = [".".join(i.split(".")[:-1]) for i in image_names1]

        self.transforms = transforms.Compose([
            # transforms.Resize((400, 200)),
            transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transforms2 = transforms.Compose([
            transforms.Resize((patch_size//2, patch_size//2)),
        ])
        self.transforms3 = transforms.Compose([
            transforms.Resize((patch_size//4, patch_size//4)),
        ])
        # self.transforms3 = transforms.CenterCrop(128,128)


        self.loader = loader
        self.labels = image_names1

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        clear_img_path = self.clear_images[index]
        moire = self.loader(moire_img_path)
        clear = self.loader(clear_img_path)
        moire = self.transforms(moire)
        clear = self.transforms(clear)


        if self.crop == True:
            i, j, h, w = transforms.RandomCrop.get_params(moire, output_size=(self.patch_size, self.patch_size))
            moire = TF.crop(moire, i, j, h, w)
            clear = TF.crop(clear, i, j, h, w)

        clear2 = self.transforms2(clear)
        clear3 = self.transforms3(clear2)
        clear_list = [clear3, clear2, clear]
        label = self.labels[index]
        return moire, clear_list, label

    def __len__(self):
        return len(self.moire_images)

# image testteststest
class FHDMI_dataset_test(Dataset):
    def __init__(self, root, crop = False, loader = default_loader,resize=None):
        self.crop = crop
        moire_data_root = os.path.join(root, 'moire')
        clear_data_root = os.path.join(root, 'clear')

        image_names1 = os.listdir(moire_data_root)
        image_names1.sort()
        self.moire_images = [os.path.join(moire_data_root, x ) for x in image_names1]

        image_names2 = os.listdir(clear_data_root)
        image_names2.sort()
        self.clear_images = [os.path.join(clear_data_root, x ) for x in image_names2]

        image_names1 = [".".join(i.split(".")[:-1]) for i in image_names1]

        self.transforms = transforms.Compose([
            transforms.Resize((1024, 1920)),
            transforms.ToTensor(),

        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # self.transforms2 = transforms.Compose([
        #     transforms.Resize((512, 512)),
        # ])
        # self.transforms3 = transforms.Compose([
        #     transforms.Resize((256, 256)),
        # ])

        self.loader = loader
        self.labels = image_names1

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        clear_img_path = self.clear_images[index]
        moire = self.loader(moire_img_path)
        clear = self.loader(clear_img_path)
        moire = self.transforms(moire)
        clear = self.transforms(clear)


        # clear2 = self.transforms2(clear)
        # clear3 = self.transforms3(clear2)

        # clear2 = self.transforms2(clear)
        # clear3 = self.transforms3(clear2)

        # clear_list = clear
        label = self.labels[index]
        return moire, clear, label

    def __len__(self):
        return len(self.moire_images)





##tipo 2018
class TIP2018moire_dataset_train(Dataset):
    def __init__(self, root,crop =True, loader = default_loader_crop):
        self.crop = crop
        moire_data_root = os.path.join(root, 'source')
        clear_data_root = os.path.join(root, 'target')

        image_names1 = os.listdir(moire_data_root)
        image_names1.sort()
        self.moire_images = [os.path.join(moire_data_root, x ) for x in image_names1]

        image_names2 = os.listdir(clear_data_root)
        image_names2.sort()
        self.clear_images = [os.path.join(clear_data_root, x ) for x in image_names2]

        image_names1 = [".".join(i.split(".")[:-1]) for i in image_names1]

        self.transforms = transforms.Compose([
            # transforms.Resize((400, 200)),
            transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transforms2 = transforms.Compose([
            transforms.Resize((128, 128)),
        ])
        self.transforms3 = transforms.Compose([
            transforms.Resize((64, 64)),
        ])
        # self.transforms3 = transforms.CenterCrop(128,128)


        self.loader = loader
        self.labels = image_names1

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        clear_img_path = self.clear_images[index]
        moire = self.loader(moire_img_path)
        clear = self.loader(clear_img_path)
        moire = self.transforms(moire)
        clear = self.transforms(clear)

        # print('clear.shape',clear.shape) # 3,504,504
        h,w = clear.shape[-2:]
        # print('hwhwhw',h,w) # 504,504

        if self.crop == True:
            i, j, h, w = transforms.RandomCrop.get_params(moire, output_size=(256, 256) )
            moire = TF.crop(moire, i, j, h, w)
            clear = TF.crop(clear, i, j, h, w)

        clear2 = self.transforms2(clear)
        clear3 = self.transforms3(clear2)
        clear_list = [clear3, clear2, clear]
        label = self.labels[index]
        return moire, clear_list, label

    def __len__(self):
        return len(self.moire_images)


##tipo 2018
class TIP2018moire_dataset_test(Dataset):
    def __init__(self, root,resize =None, loader = default_loader_crop):
        self.resize = resize
        moire_data_root = os.path.join(root, 'source')
        clear_data_root = os.path.join(root, 'target')

        image_names1 = os.listdir(moire_data_root)
        image_names1.sort()
        self.moire_images = [os.path.join(moire_data_root, x ) for x in image_names1]

        image_names2 = os.listdir(clear_data_root)
        image_names2.sort()
        self.clear_images = [os.path.join(clear_data_root, x ) for x in image_names2]

        image_names1 = [".".join(i.split(".")[:-1]) for i in image_names1]

        self.transforms = transforms.Compose([
            # transforms.Resize((400, 200)),
            transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transforms2 = transforms.Compose([
            transforms.Resize((128, 128)),
        ])
        self.transforms3 = transforms.Compose([
            transforms.Resize((64, 64)),
        ])
        # self.transforms3 = transforms.CenterCrop(128,128)


        self.loader = loader
        self.labels = image_names1

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        clear_img_path = self.clear_images[index]
        moire = self.loader(moire_img_path)
        clear = self.loader(clear_img_path)
        moire = self.transforms(moire)
        clear = self.transforms(clear)

        # print('clear.shape',clear.shape) # 3,504,504
        h,w = clear.shape[-2:]
        # print('hwhwhw',h,w) # 504,504

        if len(self.resize) == 2:
            # i, j, h, w = transforms.RandomCrop.get_params(moire, output_size=(256, 256) )
            # moire = TF.crop(moire, i, j, h, w)
            # clear = TF.crop(clear, i, j, h, w)
            moire = TF.resize(moire,self.resize)
            clear = TF.resize(clear,self.resize)
            clear2 = self.transforms2(clear)
            clear3 = self.transforms3(clear2)
        else:
            clear2,clear3=None,None
        clear_list = [clear3, clear2, clear]
        label = self.labels[index]
        return moire, clear_list, label

    def __len__(self):
        return len(self.moire_images)

def random_scale_for_pair(moire, clear, mask, is_val=False):
    if is_val == False:
        is_global = np.random.randint(0, 2)
        if is_global == 0:
            resize = transforms.Resize((256, 256))
            moire, clear,mask = resize(moire), resize(clear),resize(mask)
        else:
            resize = transforms.Resize((286, 286))
            moire, clear,mask = resize(moire), resize(clear),resize(mask)

            random_x = np.random.randint(0, moire.size[0] - 256)
            random_y = np.random.randint(0, moire.size[1] - 256)
            moire = moire.crop((random_x, random_y, random_x + 256, random_y + 256))
            clear = clear.crop((random_x, random_y, random_x + 256, random_y + 256))
            mask  = mask.crop((random_x, random_y, random_x + 256, random_y + 256))

        is_flip = np.random.randint(0, 2)
        if is_flip == 0:
            moire = moire.transpose(Image.FLIP_LEFT_RIGHT)
            clear = clear.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            pass
    else:
        resize = transforms.Resize((256, 256))
        moire, clear,mask = resize(moire), resize(clear),resize(mask)

    return moire, clear, mask

# class Moire_dataset(Dataset):
#     def __init__(self,data_path, transform=None, is_val=False, loader=default_loader):
#         self.data_path = data_path
#
#         moire_data_root = os.path.join(self.data_path, "source")
#         clear_data_root = os.path.join(self.data_path, "target")
#         mask_data_root  = os.path.join(self.data_path, "binary")
#
#         image_names = os.listdir(clear_data_root)
#         image_names = ["_".join(i.split("_")[:-1]) for i in image_names]
#
#         self.moire_images = [os.path.join(moire_data_root, x + "_source.png") for x in image_names]
#         self.clear_images = [os.path.join(clear_data_root, x + "_target.png") for x in image_names]
#         self.mask_images  = [os.path.join(mask_data_root, x + "_binary.png") for x in image_names]
#
#         self.transforms = transforms.Compose([
#             transforms.ToTensor(),
#             # transforms.Grayscale(num_output_channels=1), transforms.ToTensor()
#             #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
#
#         self.is_val = is_val
#         self.loader = loader
#         self.labels = image_names
#
#     def __getitem__(self, index):
#         label = self.labels[index]
#
#         moire_img_path = self.moire_images[index]
#         clear_img_path = self.clear_images[index]
#         mask_img_path = self.mask_images[index]
#
#         moire = self.loader(moire_img_path)
#         clear = self.loader(clear_img_path)
#         mask = self.loader(mask_img_path)
#
#         moire, clear,mask = random_scale_for_pair(moire, clear,mask, self.is_val)
#
#         moire = self.transforms(moire)
#         clear = self.transforms(clear)
#         mask = self.transforms(mask)
#
#         return moire, clear,mask, label
#
#
#     def __len__(self):
#         return len(self.moire_images)

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample




class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

class Synapse_dataset_te(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!

        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        # print(len(self.sample_list))
        return len(self.sample_list)
    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
