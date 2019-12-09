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
from utils import compute_PA, compute_MIoU

if __name__ == '__main__':

    batch_size = 1  # Batch size during training
    image_size = 512  # All images will be resized to this size using a transformer.

    # Root directory for dataset
    # data_path = '/datasets/standing_1/full'
    data_path = './datasets/standing_s_test'

    results_path = './samples_4/results_512'
    os.makedirs(results_path, exist_ok=True)

    # ckpt_path = './samples_0/checkpoint_iteration_309000.tar'
    ckpt_path = './samples_4/checkpoint_epoch=1000.tar'

    workers = 0  # Number of workers for dataloader
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation((-15, 15))
    ])

    dataset = img_transfer_alpha(root=data_path, img_size=image_size, transform=transform, is_offset=False)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=workers)

    criterion = nn.MSELoss()


    def get_layer_param(model):
        return sum([torch.numel(param) for param in model.parameters()])


    net = UNet(3, 1).to(device)

    print(net)
    print('parameters:', get_layer_param(net))
    print("Data lengths: ", len(dataset))

    print("Loading checkpoint...")

    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint['net_state_dict'])
    net.eval()

    list_loss = []
    list_psnr = []
    list_ssim = []
    list_pa = []
    list_iou = []

    print("Starting Test...")

    for i, data in enumerate(dataloader, 0):
        # -----------------------------------------------------------
        # Initial batch
        data_A = data[0].to(device)
        data_B = data[1].to(device)
        real_batch_size = data_A.size(0)

        # -----------------------------------------------------------
        # Generate fake img:
        fake_B = net(data_A)

        # -----------------------------------------------------------
        # Output training stats
        vutils.save_image(data_A, os.path.join(results_path, '%s_data_A.jpg' % str(i).zfill(6)),
                          padding=0, nrow=1, normalize=True)
        vutils.save_image(data_B, os.path.join(results_path, '%s_data_B.jpg' % str(i).zfill(6)),
                          padding=0, nrow=1, normalize=True)
        vutils.save_image(fake_B, os.path.join(results_path, '%s_fake_B.jpg' % str(i).zfill(6)),
                          padding=0, nrow=1, normalize=True)

        loss = criterion(fake_B, data_B)
        psnr = 10 * math.log10(2 ** 2 / loss.item())

        data_B_Img = Image.open(os.path.join(results_path, '%s_data_B.jpg' % str(i).zfill(6)))
        fake_B_Img = Image.open(os.path.join(results_path, '%s_fake_B.jpg' % str(i).zfill(6)))
        ssim = compare_ssim(data_B_Img, fake_B_Img)
        pa = compute_PA(data_B_Img, fake_B_Img)
        iou = compute_MIoU(data_B_Img, fake_B_Img)

        list_loss.append(loss.item())
        list_psnr.append(psnr)
        list_ssim.append(ssim)
        list_pa.append(pa)
        list_iou.append(iou)

        print('[%2d/%d]\tpsnr: %.4f\tssim: %.4f\tpa: %.4f\tiou: %.4f' % (i, len(dataloader), psnr, ssim, pa, iou))

    avg_psnr = sum(list_psnr) / len(list_psnr)
    avg_ssim = sum(list_ssim) / len(list_ssim)
    avg_pa = sum(list_pa) / len(list_pa)
    avg_miou = sum(list_iou) / len(list_iou)

    print('avg_psnr: %.4f\tavg_ssim: %.4f\tavg_pa: %.4f\tavg_iou: %.4f' % (avg_psnr, avg_ssim, avg_pa, avg_miou))

    with open(os.path.join(results_path, 'log.txt'), 'a') as f:
        for i in range(len(list_psnr)):
            f.write('[%2d/%d]\tpsnr: %.4f\tssim: %.4f\tpa: %.4f\tiou: %.4f\n' % (
                i, len(dataloader), list_psnr[i], list_ssim[i], list_pa[i], list_iou[i]))
        f.write('avg_psnr: %.4f\tavg_ssim: %.4f\tavg_pa: %.4f\tavg_iou: %.4f' % (avg_psnr, avg_ssim, avg_pa, avg_miou))

    print('Test Over')
