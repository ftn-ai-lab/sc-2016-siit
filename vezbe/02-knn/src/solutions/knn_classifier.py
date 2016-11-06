# coding=utf-8
# autor: Bojan Blagojević  sw9/2013

from math import sqrt
import operator
import warnings


class KNNClassifier(object):

    def __init__(self, k, distance='L2'):
        """
        Konstruktor KNN klasifikatora.

        :param k: koliko najblizih suseda uzeti u obzir.
        :param distance: koju udaljenost koristiti za poredjenje, moguce vrednosti su 'L1' i 'L2'
        """

        self.k = k
        self.distance = distance

    def fit(self, x, y):
        """
        :param x: podaci
        :param y: labele klasa (celobrojne vrednosti >= 0)
        :return: None
        """

        labels = list()

        for label in y:
            if label not in labels:
                labels.append(label)

        if len(labels) == self.k:
            warnings.warn("Broj klasa je jednak parametru k(broj susjeda)")

        self.x = x
        self.y = y

    @staticmethod
    def euclidean_distance(x, y):
        """
            :param x: ulazni vektor osobina jedne tačke
            :param y: ulazni vektor osobina druge tačke
            :return: euklidsko rastojanje između podatka x i podatka y
        """

        distance = 0
        for i in range(len(x)):
            distance += (x[i] - y[i])**2

        return sqrt(distance)

    @staticmethod
    def manhattan_distance(x, y):
        """
            :param x: ulazni vektor osobina jedne tačke
            :param y: ulazni vektor osobina druge tačke
            :return: menhetn rastojanje između podatka x i podatka y
        """
        distance = 0
        for i in range(len(x)):
            distance += abs(x[i] - y[i])

        return distance

    def get_k_nearest_neighbour(self, unmarked_data):
        """
            :param unmarked_data: neoznačeni podaci
            :return: k najbližih susjeda za unmarked_data
        """
        dists = {}

        for i in range(len(self.x)):
            if self.distance == "L1":
                d = self.manhattan_distance(self.x[i], unmarked_data)
            else:
                d = self.euclidean_distance(self.x[i], unmarked_data)
            dists[str(i)] = d

        # sortiranje susjeda prema udaljenosti da bi se dobilo k susjeda sa najmanjom razdaljinom
        sorted_dists = sorted(dists.iteritems(), key=operator.itemgetter(1))

        k_neighbours = []
        for i in range(self.k):
            k_neighbours.append(self.y[int(sorted_dists[i][0])])

        return k_neighbours

    def vote(self, k_neighbours):
        """
            :param k_neighbours: k najbližih susjeda
            :return: iem klase susjeda koji se najviše puta pojavljivao
        """

        votes = {}

        for label in k_neighbours:
            if label in votes:
                votes[label] += 1
            else:
                votes[label] = 1

        # sortiranje 'glasova' u opadajućem redoslijedu da bi se dobio susjed sa najviše glasova
        sorted_votes = sorted(votes.iteritems(), key=operator.itemgetter(1), reverse=True)

        return sorted_votes[0][0]

    def get_accuracy(self, results, expected_results):
        """
            :param results: vektor rezultata koji su dobijeni nakon predikcije (predviđene vrijednosti)
            :param expected_results: vektor očekivanih rezultata (tačne vrijednosti)
            :return: tačnost predikcije
        """

        num_of_equals = 0.

        for i in range(len(expected_results)):
            if results[i] == expected_results[i]:
                num_of_equals += 1

        return num_of_equals/len(results)

    def predict(self, x, expected_results=None):
        """
        :param x: podaci za klasifikaciju
        :param expected_results: vektor očekivanih rezulata
        :return: labele X podataka nakon klasifikacije
        """
        results = []

        for i in range(len(x)):
            k_neighbours = self.get_k_nearest_neighbour(x[i])
            result = self.vote(k_neighbours)
            results.append(result)

        if expected_results is not None:
            acc = self.get_accuracy(results, expected_results)
            print ("Accuracy: " + str(acc * 100) + "%")
