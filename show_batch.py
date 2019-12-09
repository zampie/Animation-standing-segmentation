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
from data_loader import img_transfer_AB, img_transfer_alpha

if __name__ == '__main__':
    # data_path = '/datasets/standing_1/full'
    # data_path = 'E:\py\image-preprocessor\_temp'
    # data_path = './datasets/standing_s'
    # data_path = './datasets/standing_test'
    data_path = './datasets/test'
    # data_path = './datasets/碧蓝幻想'

    batch_size = 16
    workers = 0
    image_size = 512

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-15, 15))
    ])

    dataset = img_transfer_alpha(root=data_path, img_size=image_size, transform=transform, is_offset=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=workers)

    print("Data lengths: ", len(dataset))
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
