# author: Aleksandar Novakovic

from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters.thresholding import threshold_adaptive
from skimage.morphology import square, erosion, dilation
from skimage.measure import label, regionprops
from skimage.exposure import rescale_intensity
from scipy.ndimage import binary_fill_holes
import numpy as np

from barcode import draw_regions

if __name__ == '__main__':
    img = imread('./../../images/snowboarders.jpg')
    img_gray = rgb2gray(img)
    p1, p2 = np.percentile(img_gray, (5, 95))
    img_gray = rescale_intensity(img_gray, in_range=(p1, p2))
    img_gray_filtered = 1. - threshold_adaptive(img_gray, block_size=39, offset=0.04)
    img_gray_filtered_erosion = erosion(img_gray_filtered, selem=square(3))
    labels = label(img_gray_filtered_erosion)
    regions = regionprops(labels)
    snowboarders = []
    for region in regions:
        bbox = region.bbox
        width = bbox[3] - bbox[1]
        height = bbox[2] - bbox[0]
        orientation = region.orientation
        if height > 15 and 1 < float(height) / width < 5 and not -0.78 < orientation < 0.85:
            snowboarders.append(region)
    snowboarders_result = draw_regions(snowboarders, img_gray_filtered_erosion.shape)
    snowboarders_result = dilation(snowboarders_result, square(5))
    snowboarders_result = binary_fill_holes(snowboarders_result)
    snowboarders_result = erosion(snowboarders_result, square(3))
    plt.imshow(snowboarders_result, 'gray')
    plt.show()