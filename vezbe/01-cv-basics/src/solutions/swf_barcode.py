"""
    @author:    SWF/2013   Dragutin Marjanovic
    @email:     dmarjanovic94@gmail.com
"""

import matplotlib.pyplot as plt

import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.filters import threshold_adaptive
from skimage.morphology import opening, square
from skimage.measure import regionprops, label


# Utility function for displaying image
def plot_image(src, gray=True):
    if gray:
        plt.imshow(src, 'gray')
    else:
        plt.imshow(src)
    plt.show()


# Utility funcion for regions drawing on pictures with given size
def draw_regions(regs, img_size):
    img_r = np.ndarray((img_size[0], img_size[1]), dtype='float32')

    for reg in regs:
        coords = reg.coords
        for coord in coords:
            img_r[coord[0], coord[1]] = 1.

    return img_r


if __name__ == '__main__':

    # Load 'barcode.jpg' image
    img_barcode = imread('../../images/barcode.jpg')

    # Show image
    # plot_image(img_barcode, False)

    # Grayscale and Adaptive threshold
    img_barcode_th = 1 - threshold_adaptive(
        image=rgb2gray(img_barcode), block_size=75, offset=.04)
    # plot_image(img_barcode_th)

    # Opening applied
    img_barcode_opening = opening(
        image=img_barcode_th, selem=square(3))

    # plot_image(img_barcode_opening)

    # Make image labeled and get regions
    img_barcode_lab = label(img_barcode_opening)
    regions = regionprops(img_barcode_lab)

    # Display histogram
    """
    ratios = []
    for region in regions:
        bbox = region.bbox
        h = bbox[2] - bbox[0]       # height
        w = bbox[3] - bbox[1]       # width
        ratios.append(float(h) / w)

    n, bins, patches = plt.hist(ratios, bins=10)
    plt.show()
    """

    # Only numbers and letters are displayed
    regions_num_letters = []
    for region in regions:
        bbox = region.bbox
        h = bbox[2] - bbox[0]       # height
        w = bbox[3] - bbox[1]       # width

        if 1.4 < float(h) / w < 10.:
            regions_num_letters.append(region)

    plot_image(src=draw_regions(regions_num_letters, img_barcode_th.shape))