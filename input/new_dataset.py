import glob
import math
import os
import random
from sys import platform

# import cv2
from PIL import Image
import numpy as np
import torch

from torch.utils.data import Dataset
import torchvision.transforms as transforms

import pyblur
import math
import cv2

def motion_blur_random(img):
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    lineLengths = [3, 5, 7, 9]
    lineTypes = ["full", "right", "left"]

    def randomAngle(kerneldim):
        kernelCenter = int(math.floor(kerneldim / 2))
        numDistinctLines = kernelCenter * 4
        validLineAngles = np.linspace(0, 180, numDistinctLines, endpoint=False)
        angleIdx = np.random.randint(0, len(validLineAngles))
        return int(validLineAngles[angleIdx])

    # motion_kernel
    lineLengthIdx = np.random.randint(0, len(lineLengths))
    lineTypeIdx = np.random.randint(0, len(lineTypes))
    lineLength = lineLengths[lineLengthIdx]
    lineType = lineTypes[lineTypeIdx]
    lineAngle = randomAngle(lineLength)
    motion_kernel = pyblur.LineKernel(lineLength, lineAngle, lineType)
    #
    img = cv2.filter2D(img, -1, motion_kernel)
    return img


def defocus_blur_random(img):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    # defocusKernelDims = [3,5,7,9]
    defocusKernelDims = [3, 5, 7]
    # kernel
    kernelidx = np.random.randint(0, len(defocusKernelDims))
    kerneldim = defocusKernelDims[kernelidx]
    defocuskernel = pyblur.DiskKernel(kerneldim)
    #
    img = cv2.filter2D(img, -1, defocuskernel)
    return img

class qxin_multiLevel_Cls_dataSet(Dataset):  # for training
    def __init__(self, imgLsts_dir,
                 img_size=224,
                 augment=False):

        super(qxin_multiLevel_Cls_dataSet, self).__init__()
        print "-"*10,"LOAD DATA","-"*10
        # --------- step1: load imgs
        imgs_withLabel = []
        lsts = sorted(glob.glob('{}/*.txt' .format(imgLsts_dir)))
        for each_lst in lsts:
            print "Loading {}...".format(each_lst),
            with open(each_lst) as f:
                for line in f:
                    imgpath,l2,l1 = line.strip().split("\t")
                    imgs_withLabel.append([imgpath,int(l2),int(l1)])
            print(", done.")

        self.imgs_withLabel = imgs_withLabel
        self.nF = len(self.imgs_withLabel)  # number of image files
        assert self.nF > 0, 'No images found in path %s' % imgLsts_dir
        print("Total {} imgs found.".format(self.nF))
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return self.nF

    def __getitem__(self, file_index):

        img_path, l2, l1 = self.imgs_withLabel[file_index]

        img = Image.open(img_path) # RGB
        if img is None:
            raise "img in None:{}".format(img_path)
            return None, None

        w, h = img.size
        scale = 224. / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        # blur aug
        if self.augment:
            rand_blur = np.random.random()
            if rand_blur < 0.3:
                img_fg = Image.fromarray(defocus_blur_random(img_fg))
            elif rand_blur < 0.6:
                img_fg = Image.fromarray(motion_blur_random(img_fg))

        # padding
        size_fg = img_fg.size

        size_bg = 280
        img_bg = Image.new("RGB", (size_bg, size_bg))
        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))

        img = img_bg

        if self.augment:
            img = transforms.RandomHorizontalFlip()(img)
            # img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(img)
            img = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)(img)
            img = transforms.RandomAffine(5, translate=(0.0, 0.0), scale=(0.95, 1.05), shear=1)(img)

        img = transforms.CenterCrop(224)(img)
        img = transforms.ToTensor()(img)

        # img_res = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and cv2 to pytorch
        # img_res = img_res.astype(np.float32)
        # img_res /= 255.0
        # img_res = torch.from_numpy(img_res)
        # qxin
        # import ipdb
        # ipdb.set_trace()
        return img, torch.tensor([l2]).item(), torch.tensor([l1]).item(),img_path


# class qxin_multiLevel_Cls_dataSet(Dataset):  # for training
#     def __init__(self, imgs_path,
#                  multiLevel_mapping="/home/img/qiuxin/data/data_qx02/cufenlei/duojifeilei_label.txt",
#                  img_size=224,
#                  augment=False):
#
#         super(qxin_multiLevel_Cls_dataSet, self).__init__()
#
#         # --------- step1: load multiLevel_mapping # demo: multiLevel_mapping /home/img/qiuxin/data/data_qx02/cufenlei/duojifeilei_label.txt
#         imgdir_to_labelid = {}
#         lines = open(multiLevel_mapping).readlines()
#         l1_to_l2 = {}
#         for each_line in lines:
#             dirname, l2, l1 = each_line.strip().split()
#             imgdir_to_labelid[dirname] = [int(x) for x in [l2, l1]]
#             if l1 not in l1_to_l2:
#                 l1_to_l2[l1] = set([l2])
#             else:
#                 l1_to_l2[l1].add(l2)
#         self.l1_to_l2 = l1_to_l2
#         # --------- step2: init image path with label
#         for r, ds, fs in os.walk(imgs_path):
#             break
#         imgs_withLabel = []
#         for each_d in sorted(ds):
#             imgs = glob.glob('%s/*.jpg' % (os.path.join(imgs_path,each_d)))
#             for each_img in imgs:
#                 imgs_withLabel.append([each_img, ] + imgdir_to_labelid[each_d])
#
#         self.imgs_withLabel = imgs_withLabel
#
#         self.nF = len(self.imgs_withLabel)  # number of image files
#         assert self.nF > 0, 'No images found in path %s' % imgs_path
#         self.img_size = img_size
#         self.augment = augment
#
#     def __len__(self):
#         return self.nF
#
#     def __getitem__(self, file_index):
#
#         img_path, l2, l1 = self.imgs_withLabel[file_index]
#         # img = cv2.imread(img_path)  # BGR
#         img = Image.open(img_path)
#         if img is None:
#             raise "img in None:{}".format(img_path)
#             return None, None
#         img = img.resize((self.img_size,self.img_size))
#
#         if self.augment:
#             img = transforms.RandomHorizontalFlip()(img)
#             img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(img)
#             img = transforms.RandomAffine(45, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)(img)
#
#
#         img = transforms.ToTensor()(img)
#
#         # img_res = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and cv2 to pytorch
#         # img_res = img_res.astype(np.float32)
#         # img_res /= 255.0
#         # img_res = torch.from_numpy(img_res)
#         # qxin
#         # import ipdb
#         # ipdb.set_trace()
#         return img, torch.tensor([l2]).item(), torch.tensor([l1]).item()

