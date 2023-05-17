# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 12:00:45 2023

@author: r.yong

Remake Online diagnosis
"""

import numpy as np
import matplotlib.pyplot as plt
from Calcul import Calcul
# import csv

# Récupérer les données Q1, Q2, H2
# file = open('test-2023-04-14_10-42-04.csv')
# csvreader = csv.reader(file)
# Q1_mes = []
# Q2_mes = []
# H_out = []
# for row in csvreader:
    # H_out.append(row[2])
    # Q1_mes.append(row[3])
    # Q2_mes.append(row[4])
# file.close()


# Paramètres
g = 9.782 # Accélération gravité
b = 422.754 # vitesse de propagation d'onde dans l'eau en m/s
D = 0.052 # diamètre intérieur du tuyau
D_boitier = 0.1 #☻ diamètre intérieur du boitier
A = np.pi * (D/2)**2 # section du tuyau
L = 57.76 # longueur du tuyau
deltaT = 1 / 1000 # pas de temps d'échantillonage
n = 10 # nombre d'échantillonage
nul = 1.005**-6 # viscosité cinématique en m^2/s
epsilon = 2.47 * 10**-4 # rugosité du PER
rho = 1000 # masse volumique de l'eau en kg/m^3

# Initialisation
Q1_init = 241.8 * 10**-3 / 60
Q2_init = 227.7 * 10**-3 / 60
H2_init = 2.5
zL_init = 13
lambda_init = 0.5*10**-6
# Variables d'état estimés après commande avant m-à-j (a priori)
Q1_moin_init = 240 * 10**-3 / 60
H2_moin_init = 2
Q2_moin_init = 220 * 10**-3 / 60
zL_moin_init = 5
lambda_moin_init = 0
x_moin = [np.array([[Q1_init],
                    [H2_init],
                    [Q2_init],
                    [zL_init],
                    [lambda_init]])]
P_moin_init = np.diag([np.abs(Q1_init-Q1_moin_init) / Q1_init, 
                       np.abs(H2_init-H2_moin_init) / H2_init,
                       np.abs(Q2_init-Q2_moin_init) / Q2_init,
                       np.abs(zL_init-zL_moin_init) / zL_init,
                       np.abs(lambda_init-lambda_moin_init) / lambda_init])
P_moin_tot = [P_moin_init]
# Matrice d'observateur
H = np.zeros((2,5))
H[0, 0] = 1
H[1, 2] = 1
# Bruit de la mesure
R = np.diag([5*10**-3, 5*10**-3])
# Gain de Kalman
K = [P_moin_init @ H.T @ np.linalg.inv(H@P_moin_init@H.T + R)]
# Simulation du signal à supprimer
# A supprimer (matrice des mesures simulées)
y = []
Q1_mes = np.abs(4*10**-3 + np.random.normal(0, 0.01**2))
Q2_mes = np.abs(3.68*10**-3 + np.random.normal(0, 0.01**2))
y.append(np.array([[Q1_mes], 
                   [Q2_mes]]))
# Variables d'état
x = [x_moin[0] + K[0]@(y[0] - H@x_moin[0])]

# Matrice Jacobienne
F_tot = []
# Covariance d'estimé
P = [(np.eye(5)-K[0]@H) * np.linalg.inv(P_moin_init)]
# P = [np.eye(5)]
# P = [np.diag([0.5, 0.1, 0.1, 0.1])]

Re_Q1 = Q1_init*D / (A*nul)
Re_Q2 = Q2_init*D / (A*nul)
frot_Q1 = 0.25 / (np.log10(epsilon/(3.7*D) + 5.74/(Re_Q1)**0.9))**2
frot_Q2 = 0.25 / (np.log10(epsilon/(3.7*D) + 5.74/(Re_Q2)**0.9))**2 # facteur de frottement
Q1_mestot = [Q1_mes]
Q2_mestot = [Q2_mes]
frot = Calcul(g, b, D, A, L, deltaT, nul, epsilon, rho)

# Boucle
for i in range(n-1):
    # Commande
    H_out = 0.75 + np.random.normal(0, 0.1) # à supprimer
    # H_in = Calcul(g, b, D, A, L, deltaT, nul, epsilon, rho)
    # H_in.press_amont(D_boitier, Q1_mes, H_out)
    H_in = 4.16 + np.random.normal(0, 0.1) # à supprimer
    u = np.array([[H_in],
                  [H_out]])
    # Simulation du signal à supprimer
    # A supprimer (matrice des mesures simulées)
    Q1_mes = np.abs(4*10**-3 + np.random.normal(0, 0.01**2))
    Q2_mes = np.abs(3.68*10**-3 + np.random.normal(0, 0.01**2))
    Q1_mestot.append(Q1_mes)
    Q2_mestot.append(Q2_mes)
    y.append(np.array([[Q1_mes], 
                       [Q2_mes]]))
    # y.append(np.array([[Q1_mes[n]], 
                       # [Q2_mes[n]]]))
    # Bruits de l'ambiance
    Q = np.diag([10**-5, 10**-2, 10**-5, 2500, 10**-6])
    # Bruit de la mesure
    R = np.diag([5*10**-3, 5*10**-3])
    # R = 5*10**-3
    # Estimées d'état à priori
    # Fonctions tièdes
    Q1_t = x[i][0] + deltaT*(-g*A*(x[i][1]-H_in)/x[i][3]- frot.coef_frot(x[i][0])*x[i][0]**2/2/D/A)
    H2_t = x[i][1] + deltaT*(-b**2/(g*A*x[i][3])*(x[i][2] - x[i][0] + x[i][4]*np.sqrt(np.abs(x[i][1]))))
    Q2_t = x[i][2] + deltaT*(-g*A*(H_out-x[i][1])/(L-x[i][3]) - frot.coef_frot(x[i][2])*x[i][2]**2/(2*D*A))
    Q1_moin = x[i][0] + deltaT/2*(-g*A*(x[i][1]-H_in)/x[i][3]
                                  - frot.coef_frot(x[i][0])*x[i][0]**2/2/D/A
                                  - g*A*(H2_t-H_in)/x[i][3]
                                  - frot.coef_frot(Q1_t)*Q1_t**2/2/D/A)
    H2_moin = x[i][1] + deltaT/2*(-b**2/(g*A*x[i][3])*(x[i][2] - x[i][0] + x[i][4]*np.sqrt(np.abs(x[i][1])))
                                  -b**2/(g*A*x[i][3])*(Q2_t - Q1_t + x[i][4]*np.sqrt(np.abs(H2_t))))
    Q2_moin = x[i][2] + deltaT/2*(-g*A*(H_out-x[i][1])/(L-x[i][3])
                                  - frot.coef_frot(x[i][2])*x[i][2]**2/(2*D*A)
                                  - g*A*(H_out-H2_t)/(L-x[i][3])
                                  - frot.coef_frot(Q2_t)*Q2_t**2/(2*D*A))
    zL_moin = x[i][3]
    # zL_moin = (2*A**2*D*g*(H_in-H_out) - frot_Q2*x[i][2]**2*L) / (frot_Q1*x[i][0]**2 - frot_Q2*x[i][2]**2)
    # lambda_moin = (x[i][0]-x[i][2]) / x [i][1]
    lambda_moin = x[i][4]
    x_moin.append(np.array([Q1_moin,
                            H2_moin,
                            Q2_moin,
                            zL_moin,
                            lambda_moin]))
    # Matrice jacobienne
    F = Calcul(g, b, D, A, L, deltaT, nul, epsilon, rho)
    F.etat(x[i], [H_in, H_out])
    F = F.matrice_F(x[i])
    F_tot.append(F)
    
    # Covariance à priori
    P_moin = F @ P[i] @ F.T + Q
    P_moin_tot.append(P_moin)
    
    # Gain de Kalman
    K.append(P_moin @ H.T @ np.linalg.inv(H@P_moin@H.T + R))
    
    # Mettre à jour
    x.append(x_moin[i+1] + K[i]@(y[i] - H@x_moin[i+1]))
    P.append((np.eye(5)-K[i]@H) * np.linalg.inv(P_moin))
    
# Graphique
# t = np.arange(0, 60, deltaT)
t = np.arange(0, n)
zL = []
Q2 = []
Q2_vrai = []
Q1 = []
Q1_vrai = []
H2 = []
H2_vrai = []
for j in x:
    zL.append(j[3][0])
    Q1.append(j[0][0])
    Q1_vrai.append(0.00403)
    Q2.append(j[2][0])
    Q2_vrai.append(3.7*10**-3)
    H2.append(j[1][0])
    H2_vrai.append(3.4)
    
plt.figure(1)
plt.title("Evolution de zL en fonction du temps")
plt.plot(t, zL, label="Estimé")
plt.legend()
plt.figure(2)
plt.title("Evolution des Débits en fonction du temps")
plt.plot(t, Q1, label="Estimé Q1")
plt.plot(t, Q2, label="Estimé Q2")
plt.plot(t, Q1_mestot, label="Mesure Q1")
plt.plot(t, Q2_mestot, label="Mesure Q2")
plt.plot(t, Q1_vrai, label="Q1 Vraie")
plt.plot(t, Q2_vrai, label="Q2 Vraie")
plt.legend()
plt.figure(3)
plt.title("Evolution de H2 en fonction du temps")
plt.plot(t, H2, label="Estimé")
plt.plot(t, H2_vrai, label="Vraie")
plt.legend()