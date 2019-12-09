import torch.utils.data as data
import torch
from PIL import Image
import math
import cv2
import matplotlib.pyplot as plt

import os
import os.path
import sys
import numpy as np
import glob
import random
import torchvision.transforms as transforms


class SampleImageFolder(data.Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.filename_list = glob.glob(os.path.join(self.root, '*'))

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):
        img = Image.open(self.filename_list[idx])

        if self.transform:
            img = self.transform(img)

        return img


class img_transfer_AB(data.Dataset):

    def __init__(self, root, transform_A=None, transform_B=None, test=False):
        self.root = root
        self.transform_A = transform_A
        self.transform_B = transform_B
        self.test = test

        self.dataset_A = []
        self.dataset_B = []

        self.walk_data(root)

    def __len__(self):
        return len(self.dataset_A)

    def __getitem__(self, idx):
        img_A = self.dataset_A[idx]
        img_B = self.dataset_B[idx]

        img_A = Image.open(img_A)
        img_B = Image.open(img_B)

        # img_A = img_A.convert('L')
        img_A = img_A.convert('RGB')

        img_B = img_B.convert('L')
        # img_B = img_B.convert('RGB')

        img_A = np.array(img_A) / 255
        img_B = np.array(img_B) / 255

        for c in range(3):
            img_A[:, :, c] *= img_B + (1 - img_B) * np.random.rand()

        img_A *= 255
        img_B *= 255

        img_A = img_A.astype('uint8')
        img_B = img_B.astype('uint8')

        img_A = Image.fromarray(img_A)
        img_B = Image.fromarray(img_B)

        if self.transform_A:
            img_A = self.transform_A(img_A)

        if self.transform_B:
            img_B = self.transform_B(img_B)

        return img_A, img_B

    def walk_data(self, root):
        if self.test:
            root_A = os.path.join(root, 'testA')
            root_B = os.path.join(root, 'testB')
        else:
            root_A = os.path.join(root, 'trainA')
            root_B = os.path.join(root, 'trainB')

        for dirpath, dirnames, filenames in os.walk(root_A):
            for filename in filenames:
                # print(filename)
                img_A = os.path.join(dirpath, filename)
                self.dataset_A.append(img_A)

        for dirpath, dirnames, filenames in os.walk(root_B):
            for filename in filenames:
                img_B = os.path.join(dirpath, filename)
                self.dataset_B.append(img_B)

        if len(self.dataset_A) != len(self.dataset_B):
            raise TypeError('Different size of A and B')


class img_transfer_alpha(data.Dataset):

    def __init__(self, root, img_size=256, transform=None, is_offset=True):
        self.root = root
        self.size = img_size
        self.transform = transform

        self.is_offset = is_offset

        self.transform_ToTensor = transforms.ToTensor()
        self.transform_Resize = transforms.Resize(self.size)
        # self.transform_RandomCrop = transforms.RandomCrop(256)
        self.transform_Norm1 = transforms.Normalize((0.5,), (0.5,))
        self.transform_Norm3 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.dataset = []

        self.walk_data(root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        img = Image.open(self.dataset[idx])
        img = img.convert('RGBA')

        if self.transform:
            img = self.transform(img)

        # --------------------------------------------------------------------
        # pad
        w, h = img.size
        img = np.array(img) / 255
        # for c in range(3):
        #     img[:, :, c] = img[:, :, c] * img[:, :, 3] + (1 - img[:, :, 3]) * np.random.rand()

        if self.is_offset:
            # offset = 0
            offset = min(w, h) // 2
            offset = random.randint(0, offset)
        else:
            offset = 0

        if h > w:
            exp_len = h - w
            if self.is_offset:
                el = random.randint(0, exp_len)
            else:
                el = exp_len // 2
            er = exp_len - el
            img = cv2.copyMakeBorder(img, offset, offset, el + offset, er + offset, cv2.BORDER_CONSTANT,
                                     value=[1., 1., 1., 0.])
        elif h < w:
            exp_len = w - h
            if self.is_offset:
                et = random.randint(0, exp_len)
            else:
                et = exp_len // 2
            eb = exp_len - et
            img = cv2.copyMakeBorder(img, et + offset, eb + offset, offset, offset, cv2.BORDER_CONSTANT,
                                     value=[1., 1., 1., 0.])

        # img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        # img = cv2.resize(img, (self.size, self.size))

        img_A = img[:, :, 0:3]
        img_B = img[:, :, 3]

        # for c in range(3):
        #     # img_A[:, :, c] = img_A[:, :, c] + (1 - img_B)  # 白色背景, 没有计算原图的透明度
        #     # img_B = (img_B > 0).astype(np.float)  #二进制透明度,BUG,与PIL的效果相同,PIL的透明度应该是二进制的
        #     img_A[:, :, c] = img_A[:, :, c] * img_B + (1 - img_B)  # 白色背景
        #     # img_A[:, :, c] = img_A[:, :, c] * img_B + (1 - img_B) * (0.9 + 0.1 * np.random.rand())  # 随机浅色背景
        #     # img_A[:, :, c] = img_A[:, :, c] * img_B + (1 - img_B) * (0.9 * np.random.rand())  # 随机深色背景
        #
        #     # img_A[:, :, c] = img_A[:, :, c] * img_B + (1 - img_B) * np.random.rand()  # 随机纯色背景
        #     # img_A[:, :, c] *= img_B + (1 - img_B) * np.random.rand()  # 随机纯色背景 （BUG, 多乘了一次A的背景）
        #     # img_A[:, :, c] *= img_B  # 黑色背景

        p = random.uniform(0, 1)

        if p < 0.4:
            for c in range(3):
                img_A[:, :, c] = img_A[:, :, c] * img_B + (1 - img_B)  # 白色背景
        elif p < 0.8:
            for c in range(3):
                img_A[:, :, c] = img_A[:, :, c] * img_B + (1 - img_B) * (0.9 + 0.1 * np.random.rand())  # 随机浅色背景

        elif p < 0.9:
            for c in range(3):
                img_A[:, :, c] = img_A[:, :, c] * img_B + (1 - img_B) * (0.9 * np.random.rand())  # 随机深色背景
        else:
            for c in range(3):
                img_A[:, :, c] *= img_B  # 黑色背景


        img_A *= 255
        img_B *= 255

        img_A = img_A.astype('uint8')
        img_B = img_B.astype('uint8')

        # _, img_B = cv2.threshold(img_B, 50, 255, cv2.THRESH_BINARY)  # mask二值化

        img_A = Image.fromarray(img_A)
        img_B = Image.fromarray(img_B)

        img_A = img_A.convert('RGB')
        img_B = img_B.convert('L')
        # --------------------------------------------------------------------
        img_A = self.transform_Resize(img_A)
        img_B = self.transform_Resize(img_B)

        img_A = self.transform_ToTensor(img_A)
        img_B = self.transform_ToTensor(img_B)

        img_A = self.transform_Norm3(img_A)
        img_B = self.transform_Norm1(img_B)

        return img_A, img_B

    def walk_data(self, root):

        for dirpath, dirnames, filenames in os.walk(root):
            for filename in filenames:
                form = os.path.splitext(filename)[-1]
                # if form != '.png':
                #     continue
                # print(filename)
                img = os.path.join(dirpath, filename)
                self.dataset.append(img)


#
# img = Image.open('E:\datasets\standing_1\\full\KanColle\ship\character_full/0015_1256.png')


def img_pro(filename, is_offset=False):
    img = Image.open(filename)
    img = img.convert('RGBA')

    # --------------------------------------------------------------------
    # pad
    w, h = img.size
    img = np.array(img) / 255
    # for c in range(3):
    #     img[:, :, c] = img[:, :, c] * img[:, :, 3] + (1 - img[:, :, 3]) * np.random.rand()

    if is_offset:
        # offset = 0
        offset = min(w, h) // 10
        offset = random.randint(0, offset)
    else:
        offset = 0

    if h > w:
        exp_len = h - w
        if is_offset:
            el = random.randint(0, exp_len)
        else:
            el = exp_len // 2
        er = exp_len - el
        img = cv2.copyMakeBorder(img, offset, offset, el + offset, er + offset, cv2.BORDER_CONSTANT,
                                 value=[1., 1., 1., 0.])
    elif h < w:
        exp_len = w - h
        if is_offset:
            et = random.randint(0, exp_len)
        else:
            et = exp_len // 2
        eb = exp_len - et
        img = cv2.copyMakeBorder(img, et + offset, eb + offset, offset, offset, cv2.BORDER_CONSTANT,
                                 value=[1., 1., 1., 0.])

    img_A = img[:, :, 0:3]
    img_B = img[:, :, 3]

    for c in range(3):
        img_A[:, :, c] = img_A[:, :, c] * img_B + (1 - img_B)  # 白色背景
        # img_A[:, :, c] = img_A[:, :, c] * img_B + (1 - img_B) * np.random.rand()  # 随机纯色背景
        # img_A[:, :, c] *= img_B + (1 - img_B) * np.random.rand()  # 随机纯色背景 （BUG, 多乘了一次A的背景）
        # img_A[:, :, c] *= img_B  # 黑色背景

    img_A *= 255
    img_B *= 255

    img_A = img_A.astype('uint8')
    img_B = img_B.astype('uint8')

    # _, img_B = cv2.threshold(img_B, 50, 255, cv2.THRESH_BINARY)  # mask二值化

    img_A = Image.fromarray(img_A)
    img_B = Image.fromarray(img_B)

    img_A = img_A.convert('RGB')
    img_B = img_B.convert('L')

    return img_A, img_B
