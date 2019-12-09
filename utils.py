from PIL import Image
import math
import cv2
import numpy as np

# data_B = Image.open('./samples_2/results_512/000000_data_B.jpg')
# fake_B = Image.open('./samples_2/results_512/000000_fake_B.jpg')
#
# data_B = data_B.convert('L')
# fake_B = fake_B.convert('L')
#
# data_B = np.array(data_B)
# fake_B = np.array(fake_B)
#
# # data_B = (data_B == 1).astype('int')
# # fake_B = (fake_B == 1).astype('int')
#
# _, data_B = cv2.threshold(data_B, 127, 255, cv2.THRESH_BINARY)  # mask二值化
# _, fake_B = cv2.threshold(fake_B, 127, 255, cv2.THRESH_BINARY)  # mask二值化
#
# pre = (data_B == fake_B).astype('int')
#
# PA = pre.sum() / pre.size
# print('%.4f' % PA)
#
# data_B = data_B.astype('float') / 255
# fake_B = fake_B.astype('float') / 255
#
# intersect = (data_B * fake_B).sum()
# union = (data_B + fake_B).astype('bool').astype('float').sum()
# MIoU = intersect / union
# print('%.4f' % MIoU)


def compute_PA(data_B, fake_B):
    data_B = data_B.convert('L')
    fake_B = fake_B.convert('L')

    data_B = np.array(data_B)
    fake_B = np.array(fake_B)

    # data_B = (data_B == 1).astype('int')
    # fake_B = (fake_B == 1).astype('int')

    _, data_B = cv2.threshold(data_B, 127, 255, cv2.THRESH_BINARY)  # mask二值化
    _, fake_B = cv2.threshold(fake_B, 127, 255, cv2.THRESH_BINARY)  # mask二值化

    pre = (data_B == fake_B).astype('int')
    PA = pre.sum() / pre.size
    return PA

def compute_MIoU(data_B, fake_B):
    data_B = data_B.convert('L')
    fake_B = fake_B.convert('L')

    data_B = np.array(data_B)
    fake_B = np.array(fake_B)

    _, data_B = cv2.threshold(data_B, 127, 255, cv2.THRESH_BINARY)  # mask二值化
    _, fake_B = cv2.threshold(fake_B, 127, 255, cv2.THRESH_BINARY)  # mask二值化

    data_B = data_B.astype('float') / 255
    fake_B = fake_B.astype('float') / 255

    intersect = (data_B * fake_B).sum()
    union = (data_B + fake_B).astype('bool').astype('float').sum()
    MIoU = intersect / union
    return MIoU


