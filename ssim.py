import math
import numpy as np
from SSIM_PIL import compare_ssim
from PIL import Image
import math
import cv2

img_A = '立ち／イージス／基本.png'
img_B = '立ち／イージス／基本／笑顔.png'

# img_A = '016400_data_A.jpg'
# img_B = '016400_data_B.jpg'

img_A = Image.open(img_A)
img_B = Image.open(img_B)

# img_A = img_A.convert('RGB')
# img_B = img_B.convert('RGB')

# --------------------------------------
# img_A = np.array(img_A) / 127.5 - 1
# img_B = np.array(img_B) / 127.5 - 1
#
# sub = (img_A - img_B)
# mse = np.multiply(sub, sub).mean()
# psnr = 10 * math.log10(2**2 / mse)

ssim = compare_ssim(img_A, img_B)

# print("mse: ", mse)
# print("psnr: ", psnr)
print("ssim: ", ssim)

