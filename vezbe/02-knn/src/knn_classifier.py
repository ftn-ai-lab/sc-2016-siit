
class KNNClassifier(object):

    def __init__(self, k, distance='L2'):
        """
        Konstruktor KNN klasifikatora.

        :param k: koliko najblizih suseda uzeti u obzir.
        :param distance: koju udaljenost koristiti za poredjenje, moguce vrednosti su 'L1' i 'L2'
        """

        # TODO: implementirati konstruktor
        # validna vrednost parametra *k* je ceo broj, veci od 0
        # validna vrednost parametra *distance* je 'L1' ili 'L2'
        pass

    def fit(self, X, y):
        """
        :param X: podaci
        :param y: labele klasa (celobrojne vrednosti >= 0)
        :return: None
        """
        # TODO: implementirati fitovanje podataka (trivijalno)
        # prikazati poruku upozorenja ako je k jednako broju klasa
        pass

    def predict(self, X):
        """
        :param X: podaci za klasifikaciju
        :return: labele X podataka nakon klasifikacije
        """
        # TODO: impementirati klasifikaciju podataka
        # (X je lista podataka, dakle nije nuzno samo jedan podatak)
        # povratna vrednost su odgovarajuce labele
        pass
