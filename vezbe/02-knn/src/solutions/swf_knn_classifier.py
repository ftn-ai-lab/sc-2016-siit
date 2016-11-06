"""
    @author:    SWF/2013   Dragutin Marjanovic
    @email:     dmarjanovic94@gmail.com
"""

import Queue
import math


class KNNClassifier(object):

    def __init__(self, k, distance='L2'):
        """
        Konstruktor KNN klasifikatora.

        :param k: koliko najblizih suseda uzeti u obzir.
        :param distance: koju udaljenost koristiti za poredjenje, moguce vrednosti su 'L1' i 'L2'
        """

        # validna vrednost parametra *k* je ceo broj, veci od 0
        # validna vrednost parametra *distance* je 'L1' ili 'L2'
        if k > 0:
            self._k = k

        self._distance = self.manhattan if 'L1' == distance else self.euclidean
        self._data = []
        self._labels = []

    def fit(self, X, y):
        """
        :param X: podaci
        :param y: labele klasa (celobrojne vrednosti >= 0)
        :return: None
        """
        # prikazati poruku upozorenja ako je k jednako broju klasa
        if self._k == len(set(y)):
            print("WARNING: Number of labels is equal to number of classes!")

        self._data = X
        self._labels = y

    def predict(self, X):
        """
        :param X: podaci za klasifikaciju
        :return: labele X podataka nakon klasifikacije
        """
        # (X je lista podataka, dakle nije nuzno samo jedan podatak)
        # povratna vrednost su odgovarajuce labele
        labels = []
        for xi in X:
            queue = Queue.PriorityQueue()
            for (datum, label) in zip(self._data, self._labels):
                queue.put((self._distance(xi, datum), label))

            labels.append(self.get_label(queue))

        return labels

    def get_label(self, queue):
        """
        :param queue: red sa sortiranim distancama
        :return: labela kojoj pripada dati test primjer
        """
        labels_list = []
        while (not queue.empty()) and (len(labels_list) < self._k):
            labels_list.append(queue.get()[1])

        return max(set(labels_list), key=labels_list.count)

    @staticmethod
    def manhattan(x, y):
        """
        :param x: tacka X
        :param y: tacka Y
        :return: udaljenost izracunata po Menhetnovom algoritmu
        """
        return sum([(xi-yi) for (xi, yi) in zip(x, y)])

    @staticmethod
    def euclidean(x, y):
        """
        :param x: tacka X
        :param y: tacka Y
        :return: udaljenost izracunata po Euklidovoj razdaljini
        """
        return math.sqrt(sum([(xi-yi)**2 for (xi, yi) in zip(x, y)]))
