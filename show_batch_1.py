import argparse
import os

import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from IPython.display import HTML
import torch.functional as F
from data_loader_1 import img_transfer_alpha
from PIL import Image

if __name__ == '__main__':
    # data_path = '/datasets/standing_1/full'
    # data_path = 'E:\py\image-preprocessor\_temp'
    # data_path = './datasets/standing'
    data_path = './datasets/test'

    batch_size = 16
    workers = 0
    image_size = 256
    transform = transforms.Compose([
        # RGB和A通道一起用默认方法resize的话会出现像素错位的情况， 图片边缘出现杂色
        # transforms.Resize(image_size),
        # transforms.Resize(image_size, interpolation=Image.BICUBIC),
        # transforms.Resize(image_size, interpolation=Image.NEAREST),
        # transforms.CenterCrop(image_size),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation((-5, 5)),
        # transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = img_transfer_alpha(root=data_path, transform=transform, is_offset=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=workers)

    print(len(dataset))
    # Plot some training images
    real_batch = next(iter(dataloader))

    plt.figure(figsize=(16, 16))
    plt.axis("off")
    plt.title("Training Images A")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0], nrow=4, padding=4, normalize=True).cpu(), (1, 2, 0)))

    plt.figure(figsize=(16, 16))
    plt.axis("off")
    plt.title("Training Images B")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[1], nrow=4, padding=4, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

    # plt.figure(figsize=(16, 16))
    # plt.axis("off")
    # plt.title("Training Images A")
    # plt.imshow(
    #     np.transpose(vutils.make_grid(real_batch[:, 0:3, :, :], nrow=4, padding=4, normalize=True).cpu(), (1, 2, 0)))
    #
    # plt.figure(figsize=(16, 16))
    # plt.axis("off")
    # plt.title("Training Images B")
    # plt.imshow(
    #     np.transpose(vutils.make_grid(real_batch[:, 3:4, :, :], nrow=4, padding=4, normalize=True).cpu(), (1, 2, 0)))
    # plt.show()
