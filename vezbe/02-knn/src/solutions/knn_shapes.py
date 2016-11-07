#  -*- coding: utf-8 -*-
# autor: Bojan Blagojević  sw9/2013


from knn_classifier import *
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
from skimage.morphology import disk
from skimage.measure import label
from skimage.measure import regionprops
from scipy.ndimage import binary_fill_holes
from skimage.morphology import erosion
from matplotlib import pyplot as pt


# utility funkcija za iscrtavanje regiona na slikama zadate velicine
def draw_regions(regs, img_size):
    img_r = np.ndarray((img_size[0], img_size[1]), dtype='float64')
    for reg in regs:
        coords = reg.coords  # coords vraca koordinate svih tacaka regiona
        for coord in coords:
            img_r[coord[0], coord[1]] = 1.
    return img_r


# funkcija koja izdvaja regione(oblike) iz slike
def prepare_image(img_path):
    img = imread(img_path)
    img_gray = rgb2gray(img)
    img_tr = 1 - img_gray

    filled_img = binary_fill_holes(img_tr).astype('float64')

    str_elem = disk(10)  # parametar je poluprecnik diska
    img_tr_er = erosion(filled_img, selem=str_elem)

    labeled_img = label(img_tr_er)
    regions = regionprops(labeled_img)

    # plt.imshow(draw_regions(regions, img.shape), 'gray')
    # plt.show()
    return regions


# dobijanje bitnih osobina za jedan region
def get_properties(shapes, training_data, labels=None):
    for i in xrange(len(shapes)):
        for region in shapes[i]:
            props = list()

            bbox = region.bbox

            h = float(bbox[2] - bbox[0])  # visina
            w = bbox[3] - bbox[1]  # sirina

            # odabrane su samo dvije osobine
            # odnos površine popunjenog prostora i cijelog regiona
            # odnos obima popunjenog prostora i cijelog regiona
            param1 = region.extent
            param2 = region.perimeter/(2*(h+w))
            props.append(param1)
            props.append(param2)

            training_data.append(props)

            if labels is not None:
                labels.append(i)
                plot_point(param1, param2, i)
            else:
                plot_point(param1, param2, None)


def plot_point(x, y, col):
    color = 'black'
    if col == 0:
        color = 'red'
    if col == 1:
        color = 'blue'
    if col == 2:
        color = 'green'
    pt.scatter(x, y, color=color)


# dobijanje svih regiona
regions_circle = prepare_image('../../data/train/circles.png')
regions_rectangle = prepare_image('../../data/train/rectangles.png')
regions_triangle = prepare_image('../../data/train/triangles.png')

shapes = [regions_circle, regions_rectangle, regions_triangle]

unmarked_regions = prepare_image('../../data/test/shapes.png')
# sortiranje regiona po x-osi da bi svaki regioni redom odgovarali klasama [0, 0, 0, 1, 1, 1, 2, 2, 2]
sorted_unm_reg = sorted(unmarked_regions, key=lambda k: k['bbox'][1])

training_data = []
labels = []
test_data = []

# pripremanje training i test podataka
get_properties(shapes, training_data, labels)
get_properties([sorted_unm_reg], test_data)

# kreiranje KNN klasifikatora uz fitovanje i predikciju
knn = KNNClassifier(1)
knn.fit(training_data, labels)
knn.predict(test_data, [0, 0, 0, 1, 1, 1, 2, 2, 2])

pt.show()
