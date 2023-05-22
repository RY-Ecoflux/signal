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
    plt.show()

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

hyp = import_données("test.csv", 1)
fs = 96000 # 1 kHz sampling frequency
T = 20 # 10s signal length
t = échelle_temps(fs, T)
# Données acoustiques
try:
    tracer_données(t, hyp)
except:
    t = t[0:len(hyp)]
    tracer_données(t, hyp)

# Low-pass filter
fc = 200
sos = scipy.signal.butter(2, fc, 'low', fs=fs, output='sos')
filtered = scipy.signal.sosfilt(sos, hyp)
plt.figure(2)
plt.plot(t, filtered)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Low-Pass at 200Hz")

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