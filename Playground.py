import numpy as np
import matplotlib.pyplot as plt


a = [1, 2, 3, 4, 5]
b = [1, 1, 1, 1, 1]


def tracer_données(x, y):
    """
    Tracer un graphique des données choisies
    Entrée : le temps et les données, liste
    Sortie : graphique
    """
    plt.figure(1)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.scatter(x, y)
    plt.xticks(rotation=45)

# def échelle_temps(fs, T):
#     """
#     Générer une séquence de temps
#     Entrée :
#         fs : fréquence d'échantillonnage, int
#         T : temps final, int
#     Sortie :
#         Echelle de temps, liste
#     """
#     t = np.arange(0, T+1/fs, 1/fs)
#     return t

try:
    tracer_données(a,b)
    plt.title("test")
    plt.show()
except:
    a = a[0:len(b)]
    tracer_données(a, b)
    plt.title("test")
    plt.show()
    
