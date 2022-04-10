from PIL import Image
import mutils as utils
import numpy as np
import os
import skimage.color as sc

def compute(dir1,dir2):
    avg_psnr, avg_ssim = 0, 0
    i = 0
    for path in os.listdir(dir1):
        h_path = 'norain-' + path.split('-')[-1]
        lgen = np.asarray(Image.open(os.path.join(dir1, path)).convert('RGB'))
        lreal = np.asarray(Image.open(os.path.join(dir2, h_path)).convert('RGB'))

        lgen = utils.quantize(sc.rgb2ycbcr(lgen)[:, :, 0])
        lreal = utils.quantize(sc.rgb2ycbcr(lreal)[:, :, 0])

        avg_psnr += utils.compute_psnr(lgen, lreal)
        avg_ssim += utils.compute_ssim(lgen, lreal)
        i = i + 1
    print("==> Valid. psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr / i, avg_ssim / i))
    return avg_psnr / i, avg_ssim / i
