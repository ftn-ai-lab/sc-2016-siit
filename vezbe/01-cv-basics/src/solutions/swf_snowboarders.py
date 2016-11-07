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
    
    # Load 'snowboarders.jpg' image
    img_snow = imread('../../images/snowboarders.jpg')

    # Show image
    plot_image(img_snow, False)

    # Grayscale and Adaptive threshold
    img_snow_th = 1-threshold_adaptive(
        image=rgb2gray(img_snow), block_size=67, offset=.07)
    plot_image(img_snow_th)

    # Opening applied
    img_snow_opening = opening(
        image=img_snow_th, selem=square(4))

    plot_image(img_snow_opening)

    # Make image labeled and get regions
    img_snow_lab = label(img_snow_opening)
    regions = regionprops(img_snow_lab)

    # Only numbers and letters are displayed
    regions_snowborders = []
    for region in regions:
        bbox = region.bbox
        h = bbox[2] - bbox[0]       # height
        w = bbox[3] - bbox[1]       # width

        # Comes down to regions' parameters adjusting
        if (float(h) / w > .8 and h > 20) or h > 30:
            regions_snowborders.append(region)

    print(len(regions_snowborders))
    plot_image(src=draw_regions(regions_snowborders, img_snow_th.shape))