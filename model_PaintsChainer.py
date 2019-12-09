import torch
import torch.nn as nn
import torch.functional as F


class UNet(nn.Module):
    def __init__(self, in_c=3, out_c=3):
        super(UNet, self).__init__()
        self.down0 = nn.Conv2d(in_c, 32, 3, 1, 1)
        self.down1 = nn.Conv2d(32, 64, 4, 2, 1)
        self.down2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.down3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.down4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.down5 = nn.Conv2d(128, 256, 4, 2, 1)
        self.down6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.down7 = nn.Conv2d(256, 512, 4, 2, 1)
        self.down8 = nn.Conv2d(512, 512, 3, 1, 1)

        self.up8 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.up7 = nn.Conv2d(512, 256, 3, 1, 1)
        self.up6 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.up5 = nn.Conv2d(256, 128, 3, 1, 1)
        self.up4 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.up3 = nn.Conv2d(128, 64, 3, 1, 1)
        self.up2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.up1 = nn.Conv2d(64, 32, 3, 1, 1)
        self.up0 = nn.ConvTranspose2d(64, out_c, 3, 1, 1)

        self.bn_down0 = nn.BatchNorm2d(32)
        self.bn_down1 = nn.BatchNorm2d(64)
        self.bn_down2 = nn.BatchNorm2d(64)
        self.bn_down3 = nn.BatchNorm2d(128)
        self.bn_down4 = nn.BatchNorm2d(128)
        self.bn_down5 = nn.BatchNorm2d(256)
        self.bn_down6 = nn.BatchNorm2d(256)
        self.bn_down7 = nn.BatchNorm2d(512)
        self.bn_down8 = nn.BatchNorm2d(512)

        self.bn_up8 = nn.BatchNorm2d(512)
        self.bn_up7 = nn.BatchNorm2d(256)
        self.bn_up6 = nn.BatchNorm2d(256)
        self.bn_up5 = nn.BatchNorm2d(128)
        self.bn_up4 = nn.BatchNorm2d(128)
        self.bn_up3 = nn.BatchNorm2d(64)
        self.bn_up2 = nn.BatchNorm2d(64)
        self.bn_up1 = nn.BatchNorm2d(32)

    def forward(self, x):
        d0 = torch.relu(self.bn_down0(self.down0(x)))
        d1 = torch.relu(self.bn_down1(self.down1(d0)))
        d2 = torch.relu(self.bn_down2(self.down2(d1)))
        d3 = torch.relu(self.bn_down3(self.down3(d2)))
        d4 = torch.relu(self.bn_down4(self.down4(d3)))
        d5 = torch.relu(self.bn_down5(self.down5(d4)))
        d6 = torch.relu(self.bn_down6(self.down6(d5)))
        d7 = torch.relu(self.bn_down7(self.down7(d6)))
        d8 = torch.relu(self.bn_down8(self.down8(d7)))

        u8 = torch.relu(self.bn_up8(self.up8(torch.cat((d7, d8), 1))))
        u7 = torch.relu(self.bn_up7(self.up7(u8)))
        u6 = torch.relu(self.bn_up6(self.up6(torch.cat((d6, u7), 1))))
        u5 = torch.relu(self.bn_up5(self.up5(u6)))
        u4 = torch.relu(self.bn_up4(self.up4(torch.cat((d4, u5), 1))))
        u3 = torch.relu(self.bn_up3(self.up3(u4)))
        u2 = torch.relu(self.bn_up2(self.up2(torch.cat((d2, u3), 1))))
        u1 = torch.relu(self.bn_up1(self.up1(u2)))
        out = self.up0(torch.cat((d0, u1), 1))

        return out


# class DIS(chainer.Chain):
#     def __init__(self):
#         super(DIS, self).__init__(
#                 c1 = L.Convolution2D(3, 32, 4, 2, 1),
#                 c2 = L.Convolution2D(32, 32, 3, 1, 1),
#                 c3 = L.Convolution2D(32, 64, 4, 2, 1),
#                 c4 = L.Convolution2D(64, 64, 3, 1, 1),
#                 c5 = L.Convolution2D(64, 128, 4, 2, 1),
#                 c6 = L.Convolution2D(128, 128, 3, 1, 1),
#                 c7 = L.Convolution2D(128, 256, 4, 2, 1),
#                 l8l = L.Linear(None, 2, wscale=0.02*math.sqrt(8*8*256)),
#
#                 bnc1 = L.BatchNormalization(32),
#                 bnc2 = L.BatchNormalization(32),
#                 bnc3 = L.BatchNormalization(64),
#                 bnc4 = L.BatchNormalization(64),
#                 bnc5 = L.BatchNormalization(128),
#                 bnc6 = L.BatchNormalization(128),
#                 bnc7 = L.BatchNormalization(256),
#         )
#
#     def calc(self,x, test = False):
#         h = F.relu(self.bnc1(self.c1(x), test=test))
#         h = F.relu(self.bnc2(self.c2(h), test=test))
#         h = F.relu(self.bnc3(self.c3(h), test=test))
#         h = F.relu(self.bnc4(self.c4(h), test=test))
#         h = F.relu(self.bnc5(self.c5(h), test=test))
#         h = F.relu(self.bnc6(self.c6(h), test=test))
#         h = F.relu(self.bnc7(self.c7(h), test=test))
#         return  self.l8l(h)