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
import torch.functional as F
from unet_model import UNet
# from model_PaintsChainer import UNet
from data_loader import img_transfer_AB, img_transfer_alpha
import math
from SSIM_PIL import compare_ssim
from PIL import Image
import time
from data_loader import img_pro

if __name__ == '__main__':
    # filename = './EE9U_HiU8AAeWON.jpg'
    # filename = './75853858_p0.png'
    # filename = './74419197_p0.png'
    filename = './75751162_p0.png'
    # filename = './64682730_p0.png'

    img = Image.open(filename)
    img = img.convert('RGBA')

    ckpt_path = './samples_5/checkpoint_epoch=1000.tar'
    # ckpt_path = './samples_0/checkpoint_iteration_154500.tar'

    # device = torch.device("cpu")
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    transform = transforms.Compose([
        transforms.Resize(512),
        # transforms.CenterCrop(256),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation((-15, 15))
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
    ])

    img = transform(img)
    img = img.unsqueeze(0)

    criterion = nn.MSELoss()


    def get_layer_param(model):
        return sum([torch.numel(param) for param in model.parameters()])


    net = UNet(3, 1).to(device)

    print(net)
    print('parameters:', get_layer_param(net))
    print("Loading checkpoint...")

    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint['net_state_dict'])
    net.eval()

    str_time = time.time()
    print("Starting Test...")

    # -----------------------------------------------------------
    # Initial batch
    data_A = img[:, 0:3, :, :].to(device)
    data_B = img[:, 3:4, :, :].to(device)

    # -----------------------------------------------------------
    # Generate fake img:
    fake_B = net(data_A)
    loss = criterion(fake_B, data_B)
    psnr = 10 * math.log10(2 ** 2 / loss.item())

    # -----------------------------------------------------------
    # Output training stats
    vutils.save_image(data_A, os.path.join('./', '%s_data_A.jpg' % filename[0:-4]),
                      padding=0, nrow=1, normalize=True)
    vutils.save_image(data_B, os.path.join('./', '%s_data_B.jpg' % filename[0:-4]),
                      padding=0, nrow=1, normalize=True)
    vutils.save_image(fake_B, os.path.join('./', '%s_fake_B.jpg' % filename[0:-4]),
                      padding=0, nrow=1, normalize=True)

    data_B = Image.open(os.path.join('./', '%s_data_B.jpg' % filename[0:-4]))
    fake_B = Image.open(os.path.join('./', '%s_fake_B.jpg' % filename[0:-4]))
    ssim = compare_ssim(data_B, fake_B)

    print('psnr: %.4f\tssim: %.4f' % (psnr, ssim))

    print('Test Over, cost %.4fs:' % (time.time() - str_time))
