# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:27:55 2023

@author: r.yong

Window function test
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import csv


def import_données(nom_fich, nb_col, nb_col2=None):
    """
    Ouvrir le fichier CSV
    Entrée : 
        nom du fichier CSV, sous forme .csv, str
        nombre de colonne de données à récupérer, int
    Sortie : 
        colonne de données, liste
    """
    file = open(nom_fich)
    csvreader = csv.reader(file)
    données = []
    for row in csvreader:
        données.append(row[nb_col])
    file.close()
    return données

def tracer_données(n, x, y):
    """
    Tracer un graphique des données choisies
    Entrée : 
        nombre du figure, int
        le temps, liste de float
        les données, liste de float
    Sortie : 
        graphique
    """
    plt.figure(n)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    # Par défaut, l'index 0 est le titre de la colonne
    plt.scatter(x[1:], y[1:])
    plt.xticks(rotation=45)

hyp = import_données("test.csv", 1)
fs = 96000 # 1 kHz sampling frequency
T = 20 # 10s signal length
t = np.arange(0, T+1/fs, 1/fs)
# Données acoustiques
try:
    tracer_données(1, t, hyp)
    plt.title("Données brutes de l'hydrophone")
    plt.show()
except:
    t = t[0:len(hyp)]
    tracer_données(1, t, hyp)
    plt.title("Données brutes de l'hydrophone")
    plt.show()

# Low-pass filter
fc = 200
sos = scipy.signal.butter(2, fc, 'low', fs=fs, output='sos')
filtered = scipy.signal.sosfilt(sos, hyp)
# Données acoustiques filtrées
tracer_données(2, t, filtered)
plt.title("Passe-Bas à 200Hz")
plt.show()

# Hanning window
m = t.size
sig_win = filtered * np.hanning(m)
plt.figure(3)
plt.plot(t, 20*np.log10(np.abs(sig_win)))
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [dB]")
plt.title("Hann Window Applied")

# Power spectral analysis
# f contains the frequency components
# S is the PSD
(f, S) = scipy.signal.periodogram(sig_win, fs, scaling='density')
plt.figure(4)
plt.semilogy(f, S)
# plt.ylim([1e-10, 1e2])
plt.xlim([0,300])
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title("Power Spectral Analysis")
plt.show()