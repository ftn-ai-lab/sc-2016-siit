
class NaiveBayesClassifier(object):
"""
Naivni Bayes klasifikator za višedimenzionalne podatke sa kontinualnim vrednostima.
"""

    def __init__(self):
        """
        Konstruktor Naivnog Bayes klasifikatora.
        """
        pass

    def fit(self, X, y):
        """
        :param X: ulazni podaci (lista vektora)
        :param y: labele klasa (celobrojne vrednosti >= 0)
        :return: None
        """
        # TODO: implementirati fitovanje podataka
        # 1. izracunati verovatnoce za svaku klasu P(y0), P(y1)...
        # 2. na osnovu y izdvojiti podatke za svaku klasu
        # 3. za svaku klasu izracunati srednje vrednosti i kovarijansu matricu
        pass

    def predict(self, X):
        """
        :param X: podaci za klasifikaciju
        :return: labele X podataka nakon klasifikacije
        """
        # TODO: impementirati klasifikaciju podataka
        # Iskoristiti Bayesovu teoremu
        # povratna vrednost su odgovarajuce labele za svaki podatak u X
        pass
