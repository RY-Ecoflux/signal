import numpy as np


def échelle_temps(fs, T):
    """
    Générer une séquence de temps
    Entrée :
        fs : fréquence d'échantillonnage, int
        T : temps final, int
    Sortie :
        Echelle de temps, liste
    """
    t = np.arange(0, T+1/fs, 1/fs)
    return t

print(échelle_temps(10, 5))