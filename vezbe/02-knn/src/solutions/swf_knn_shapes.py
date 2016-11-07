"""
    @author:    SWF/2013   Dragutin Marjanovic
    @email:     dmarjanovic94@gmail.com
"""

from swf_knn_classifier import KNNClassifier
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
from skimage.morphology import disk
from skimage.measure import label
from skimage.measure import regionprops
from scipy.ndimage import binary_fill_holes
from skimage.morphology import erosion
import matplotlib.pyplot as plt


# Utility funcion for regions drawing on pictures with given size
def draw_regions(regs, img_size):
    img_r = np.ndarray((img_size[0], img_size[1]), dtype='float32')

    for reg in regs:
        coords = reg.coords
        for coord in coords:
            img_r[coord[0], coord[1]] = 1.

    return img_r


# Get regions from image
def get_image_regions(img_path):
    # Make threshold of gray image
    img_tr = 1 - rgb2gray(imread(img_path))

    # Fill holes
    img_filled = binary_fill_holes(img_tr).astype('float32')

    str_elem = disk(10)  # parametar je poluprecnik diska
    img_tr_er = erosion(img_filled, selem=str_elem)

    img_labeled = label(img_tr_er)
    return regionprops(img_labeled)


def get_properties(all_shapes, training=True):
    data = []
    lbls = []
    lbl = 0
    for regions in all_shapes:
        for region in regions:
            bbox = region.bbox
            h = float(bbox[2] - bbox[0])
            w = float(bbox[3] - bbox[1])

            data.append([region.extent, region.perimeter/(2*(h+w))])

            if lbls is not None:
                lbls.append(lbl)

        lbl += 1

    if training:
        return data, lbls
    else:
        return data

if __name__ == "__main__":
    regions_circle = get_image_regions('../../data/train/circles.png')
    regions_rectangle = get_image_regions('../../data/train/rectangles.png')
    regions_triangle = get_image_regions('../../data/train/triangles.png')

    shapes = [regions_circle, regions_rectangle, regions_triangle]

    unmarked_regions = get_image_regions('../../data/test/shapes.png')

    sorted_unm_reg = sorted(unmarked_regions, key=lambda k: k['bbox'][1])

    training_data, labels = get_properties(shapes)
    test_data = get_properties([sorted_unm_reg], False)

    knn = KNNClassifier(1)
    knn.fit(training_data, labels)

    # Our results
    results = knn.predict(test_data)
    real_labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    true_predicts = 0
    for lt, lr in zip(results, real_labels):
        if lt == lr:
            true_predicts += 1

    print "{0:.2f}% of true predicts!".format(float(true_predicts)/len(results) * 100)
