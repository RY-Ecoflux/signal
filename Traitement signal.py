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

file = open("test.csv")
csvreader = csv.reader(file)
hyp = []
for row in csvreader:
    hyp.append(row[1])
file.close()

print(len(hyp))
fs = 96000 # 1 kHz sampling frequency
T = 20 # 10s signal length
t = np.arange(0, 1611073)
plt.figure(1)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.plot(t, hyp)

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