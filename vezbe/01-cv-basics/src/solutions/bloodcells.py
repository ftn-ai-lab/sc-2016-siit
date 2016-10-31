# author: Aleksandar Novakovic

from skimage.io import imread
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
from skimage.filters.thresholding import threshold_adaptive
from skimage.morphology import square, erosion
from skimage.exposure import rescale_intensity
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
import numpy as np

from barcode import draw_regions


if __name__ == '__main__':
    img = imread('./../../images/bloodcells.jpg')

    p1, p2 = np.percentile(img, (0, 90))
    img = rescale_intensity(img, in_range=(p1, p2))
    img_gray = rgb2gray(img)
    img_gray_filtered = 1. - threshold_adaptive(img_gray, block_size=31, offset=0.02)
    img_gray_filtered = erosion(img_gray_filtered, selem=square(3))
    filled = binary_fill_holes(img_gray_filtered)
    labels = label(filled)
    regions = regionprops(labels)
    cells = []
    for region in regions:
        bbox = region.bbox
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]
        if 20 < height and 10 < width:
            cells.append(region)
    print "Number if cells:", len(cells)
    cells_result = draw_regions(cells, img_gray.shape)
    plt.imshow(cells_result, 'gray')
    plt.show()