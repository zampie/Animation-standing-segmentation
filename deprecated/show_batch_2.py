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
from IPython.display import HTML
import torch.functional as F
from data_loader import img_transfer_AB, img_transfer_alpha

if __name__ == '__main__':
    # data_path = '/datasets/standing_1/full'
    # data_path = 'E:\py\image-preprocessor\_temp'
    data_path = 'E:\py\image-preprocessor\KanColle'

    batch_size = 16
    workers = 0
    image_size = 256

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        # transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
    ])

    dataset = img_transfer_alpha(root=data_path, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=workers)

    print(len(dataset))
    # Plot some training images
    real_batch = next(iter(dataloader))

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Training Images A")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[:, 0:3], nrow=4, padding=4, normalize=True).cpu(), (1, 2, 0)))

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Training Images B")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[:, 3:4], nrow=4, padding=4, normalize=True).cpu(), (1, 2, 0)))
    plt.show()
