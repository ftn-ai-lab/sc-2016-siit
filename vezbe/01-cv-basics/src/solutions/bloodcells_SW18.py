# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 23:24:54 2016

@author: Sebastijan Stevanovic      SW 18/2013
"""


import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters.rank import threshold
from skimage.morphology import opening,disk
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops

def draw_regions(regs, img_size):
    img_r = np.ndarray((img_size[0], img_size[1]), dtype='float32')
    for reg in regs:
        coords = reg.coords
        for coord in coords:
            img_r[coord[0], coord[1]] = 1.
    return img_r

if __name__ == '__main__':
    img = imread('../../images/bloodcells.jpg')
    img_gray = rgb2gray(img)
    
    #lokalni threshold
    local_size = np.ones((11, 11), dtype='uint8')
    img_tr = threshold(1-img_gray, local_size)

    #opening
    img_tr_op = opening(img_tr, selem=disk(3))
    
    #punjenje krugova
    img_filled = binary_fill_holes(img_tr_op)
    
    #izdvajamo/obelezavamo regione 
    labeled_img = label(img_filled) 
    regions = regionprops(labeled_img)
    
    #sa histograma mozemo zakljuciti da regioni koji imaju 
    #povrsinu vecu od otprilike 270 predstavljaju crvena krvna zrnca
    ratios = []
    for region in regions:
        ratios.append(region.area)
    n, bins, patches = plt.hist(ratios, bins=10)
    plt.show()

    regions_cells = []
    for region in regions:
        if region.area > 270:
            regions_cells.append(region)

    plt.imshow(draw_regions(regions_cells, img_filled.shape), 'gray')
    print("Broj crvenih krvnih zrnaca:",len(regions_cells))



