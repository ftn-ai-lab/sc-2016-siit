"""
    @author:    SWF/2013   Dragutin Marjanovic
    @email:     dmarjanovic94@gmail.com
"""

import matplotlib.pyplot as plt

import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.filters import threshold_adaptive
from skimage.morphology import opening, disk
from skimage.measure import regionprops, label

from scipy import ndimage as ndi


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

    # Load 'blood.jpg' image
    img_blood = imread('../../images/bloodcells.jpg')

    # Show image
    plot_image(img_blood, False)

    # Grayscale and Adaptive threshold
    img_blood_th = 1 - threshold_adaptive(
        image=rgb2gray(img_blood), block_size=31, offset=.01)
    plot_image(img_blood_th)

    # Opening applied
    img_blood_opening = opening(
        image=img_blood_th, selem=disk(2))

    img_filled = ndi.binary_fill_holes(img_blood_opening)
    # plot_image(img_blood_opening)
    plot_image(img_filled)

    # Make image labeled and get regions
    img_blood_lab = label(img_filled)
    regions = regionprops(img_blood_lab)

    # Only numbers and letters are displayed
    regions_blood = []
    for region in regions:
        bbox = region.bbox
        h = bbox[2] - bbox[0]  # height
        w = bbox[3] - bbox[1]  # width

        # if (float(h) / w > .8 and h > 20) or h > 30:
        #    regions_bloodborders.append(region)
        # print region.centroid
        # print region.area
        # print region.perimeter

    print(len(regions_blood))
    plot_image(src=draw_regions(regions_blood, img_blood_th.shape))