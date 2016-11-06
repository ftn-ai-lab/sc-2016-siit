# author: Aleksandar Novakovic

from skimage.io import imread
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.filters.thresholding import threshold_adaptive
from skimage.morphology import opening, square
from skimage.measure import label, regionprops

def draw_regions(regs, img_size):
    img = np.ndarray((img_size[0], img_size[1]), dtype='float64')
    for reg in regs:
        coords = reg.coords
        for coord in coords:
            img[coord[0]][coord[1]] = 1.
    return img

if __name__ == '__main__':
    barcode_img = imread('./../../images/barcode.jpg')
    barcode_gray = rgb2gray(barcode_img)
    barcode_filtered = 1. - threshold_adaptive(barcode_gray, block_size=75, offset=0.04)
    barcode_filtered = opening(barcode_filtered, selem=square(3))
    labels = label(barcode_filtered)
    regions = regionprops(labels)
    num_regions = []
    for region in regions:
        bbox = region.bbox
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]
        if 25 < height < 45 and 1 <= float(height) / width <= 6 and (region.extent < 0.57 or region.extent > 0.80):
            num_regions.append(region)
    barcode_result = draw_regions(num_regions, barcode_gray.shape)
    plt.imshow(barcode_result, 'gray')
    plt.show()
