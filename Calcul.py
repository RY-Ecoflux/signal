# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:48:36 2023

@author: r.yong
"""
import numpy as np
from scipy.misc import derivative  # Fonction de dérivation


# Fonctions par méthode de Heun
class Calcul():
    """Dérivées partielles des fonctions exprimant les variables d'état, selon la méthode Heun"""
   
    def __init__(self, g, b, D, A, L, deltaT, nul, epsilon, rho):
        self.g = g # Accélération gravité
        self.b = b # vitesse de propagation d'onde dans l'eau en m/s
        self.D = D # diamètre intérieur du tuyau
        self.A = A # section du tuyau
        self.L = L # longueur du tuyau
        self.deltaT = deltaT # pas de temps d'échantillonage
        self.nul = nul # viscosité cinématique en m^2/s
        self.epsilon = epsilon # rugosité du PER
        self.rho = rho # masse volumique de l'eau en kg/m^3
                   
    def etat(self, x, u):
        self.x = x
        self.H_in = u[0]
        self.H_out = u[1]
    
    def coef_frot(self, débit):
        """Calculer le coefficient de frottement en fonction du débit
        Entrée : débit (integer)
        Sortie : coefficient de frottement (integer)"""
        #print(f"I am débit: {débit}")
        Re = débit*self.D / (self.A*self.nul)
        return 0.25 / (np.log10(self.epsilon/(3.7*self.D) + 5.74/(Re)**0.9))**2
    
    def press_amont(self, D2, Q1, H2):
        """
        Calculer la pression en amont
        Entrée :
            D2 : diamètre du boitier en m
            Q1 : débit volumique en amont en m^3/s
            H2 : pression mesurée en aval en m
        Returns
            H1 : pression en amont en m

        """
        self.mu = self.nul * self.rho # viscosité dyn en kg / (m.s)
        Re1 = Q1*self.D / (self.A*self.nul)
        U1 = Q1 / self.A
        k = (1-(self.D/D2)**2)**2 # coef de perte
        delta_hs = k * U1**2 / 2 / self.g # perte de charge singulière
        lamda = (100*Re1) ** -0.25
        delta_hl = lamda * (self.L/self.D) * (U1**2) / 2 / self.g # perte de charge linéaire
        delta_htot = delta_hs + delta_hl # perte de pression totale
        return H2 + delta_htot
        
    def fQ1dQ1(self, Q1):
        """Construire une fonction Q1 à dériver pour composer la matrice F
        Entrée :
            Rien
        Sortie : 
            Fonction Q1 à dériver par rapport à Q1"""
        #print(f"I am vecteur d'état: {self.x[1]}")
        Q1_t = Q1 + self.deltaT*(-self.g*self.A*(self.x[1]-self.H_in)/self.x[3]
                                 - self.coef_frot(Q1)*Q1**2/2/self.D/self.A)
        #print(f"I am Q1: {Q1}")
        H2_t = self.x[1] + self.deltaT*(-self.b**2/(self.g*self.A*self.x[3])*(self.x[2] - Q1
                                                                              + self.x[4]*np.sqrt(np.abs(self.x[1]))))
        return Q1 + self.deltaT/2*(-self.g*self.A*(self.x[1]-self.H_in)/self.x[3]
                                   - self.coef_frot(Q1)*Q1**2/2/self.D/self.A
                                   - self.g*self.A*(H2_t-self.H_in)/self.x[3]
                                   - self.coef_frot(Q1_t)
                                   *Q1_t**2/2/self.D/self.A)
    
    def fH2dQ1(self, Q1):
        """Construire une fonction H2 à dériver par rapport au Q1
        Entrée :
            Rien
        Sortie : 
            Fonction H2 à dériver par rapport au Q1"""
        Q1_t = Q1 + self.deltaT*(-self.g*self.A*(self.x[1]-self.H_in)/self.x[3]
                                 - self.coef_frot(Q1)*Q1**2/2/self.D/self.A)
        H2_t = self.x[1] + self.deltaT*(-self.b**2/(self.g*self.A*self.x[3])*(self.x[2] - Q1
                                                                              + self.x[4]*np.sqrt(np.abs(self.x[1]))))
        Q2_t = self.x[2] + self.deltaT*(-self.g*self.A*(self.H_out-self.x[1])/(self.L-self.x[3])
                                        - self.coef_frot(self.x[2])*self.x[2]**2/(2*self.D*self.A))
        return self.x[1] + self.deltaT/2*(-self.b**2/(self.g*self.A*self.x[3])*(self.x[2] - Q1
                                                                                + self.x[4]*np.sqrt(np.abs(self.x[1])))
                                          -self.b**2/(self.g*self.A*self.x[3])*(Q2_t - Q1_t
                                                                                + self.x[4]*np.sqrt(np.abs(H2_t))))
    def fQ2dQ1(self, Q1):
        """Construire une fonction Q2 à dériver par rapport au Q1
        Entrée :
            Rien
        Sortie : 
            Fonction Q2 à dériver par rapport au Q1"""
        H2_t = self.x[1] + self.deltaT*(-self.b**2/(self.g*self.A*self.x[3])*(self.x[2] - Q1
                                                                         + self.x[4]*np.sqrt(np.abs(self.x[1]))))
        Q2_t = self.x[2] + self.deltaT*(-self.g*self.A*(self.H_out-self.x[1])/(self.L-self.x[3])
                                        - self.coef_frot(self.x[2])*self.x[2]**2/(2*self.D*self.A))
        return self.x[2] + self.deltaT/2*(-self.g*self.A*(self.H_out-self.x[1])/(self.L-self.x[3])
                                          - self.coef_frot(self.x[2])*self.x[2]**2/2/self.D/self.A
                                          - self.g*self.A*(self.H_out-H2_t)/(self.L-self.x[3])
                                          - self.coef_frot(Q2_t)*Q2_t**2/2/self.D/self.A)
    
    def fQ1dH2(self, H2):
        """Construire une fonction Q1 à dériver par rapport au H2
        Entrée :
            Rien
        Sortie : 
            Fonction Q2 à dériver par rapport au H2"""
        Q1_t = self.x[0] + self.deltaT*(-self.g*self.A*(H2-self.H_in)/self.x[3]
                                        - self.coef_frot(self.x[0])*self.x[0]**2/2/self.D/self.A)
        H2_t = H2 + self.deltaT*(-self.b**2/(self.g*self.A*self.x[3])*(self.x[2] - self.x[0]
                                                                       + self.x[4]*np.sqrt(np.abs(H2))))
        return self.x[0] + self.deltaT/2*(-self.g*self.A*(H2-self.H_in)/self.x[3]
                                          - self.coef_frot(self.x[0])*self.x[0]**2/2/self.D/self.A
                                          -self.g*self.A*(H2_t-self.H_in)/self.x[3]
                                          - self.coef_frot(Q1_t)
                                          *Q1_t**2/2/self.D/self.A)
    
    def fH2dH2(self, H2):
        """Construire une fonction H2 à dériver par rapport au H2
        Entrée :
            Rien
        Sortie : 
            Fonction H2 à dériver par rapport au Q1"""
        Q1_t = self.x[0] + self.deltaT*(-self.g*self.A*(H2-self.H_in)/self.x[3]
                                        - self.coef_frot(self.x[0])*self.x[0]**2/2/self.D/self.A)
        H2_t = H2 + self.deltaT*(-self.b**2/(self.g*self.A*self.x[3])*(self.x[2] - self.x[0] 
                                                                       + self.x[4]*np.sqrt(np.abs(H2))))
        Q2_t = self.x[2] + self.deltaT*(-self.g*self.A*(self.H_out-H2)/(self.L-self.x[3])
                                        - self.coef_frot(self.x[2])*self.x[2]**2/(2*self.D*self.A))
        return H2 + self.deltaT/2*(-self.b**2/(self.g*self.A*self.x[3])*(self.x[2] - self.x[0]
                                                                         + self.x[4]*np.sqrt(np.abs(H2)))
                                   -self.b**2/(self.g*self.A*self.x[3])*(Q2_t - Q1_t
                                                                         + self.x[4]*np.sqrt(np.abs(H2_t))))
    
    def fQ2dH2(self, H2):
        """Construire une fonction Q2 à dériver par rapport au H2
        Entrée :
            Rien
        Sortie : 
            Fonction Q2 à dériver par rapport au H2"""
        H2_t = H2 + self.deltaT*(-self.b**2/(self.g*self.A*self.x[3])*(self.x[2] - self.x[0]
                                                                       + self.x[4]*np.sqrt(np.abs(H2))))
        Q2_t = self.x[2] + self.deltaT*(-self.g*self.A*(self.H_out-H2)/(self.L-self.x[3])
                                        - self.coef_frot(self.x[2])*self.x[2]**2/(2*self.D*self.A))
        return self.x[2] + self.deltaT/2*(-self.g*self.A*(self.H_out-H2)/(self.L-self.x[3])
                                          - self.coef_frot(self.x[2])*self.x[2]**2/2/self.D/self.A
                                          - self.g*self.A*(self.H_out-H2_t)/(self.L-self.x[3])
                                          - self.coef_frot(Q2_t)*Q2_t**2/2/self.D/self.A)
    
    def fQ1dQ2(self, Q2):
        """Construire une fonction Q1 à dériver par rapport au Q2
        Entrée :
            Rien
        Sortie : 
            Fonction Q1 à dériver par rapport à Q2"""
        Q1_t = self.x[0] + self.deltaT*(-self.g*self.A*(self.x[1]-self.H_in)/self.x[3]
                                        - self.coef_frot(self.x[0])*self.x[0]**2/2/self.D/self.A)
        H2_t = self.x[1] + self.deltaT*(-self.b**2/(self.g*self.A*self.x[3])*(Q2 - self.x[0]
                                                                              + self.x[4]*np.sqrt(np.abs(self.x[1]))))
        return self.x[0] + self.deltaT/2*(-self.g*self.A*(self.x[1]-self.H_in)/self.x[3]
                                          - self.coef_frot(self.x[0])*self.x[0]**2/2/self.D/self.A
                                          - self.g*self.A*(H2_t-self.H_in)/self.x[3]
                                          - self.coef_frot(Q1_t)*Q1_t**2/2/self.D/self.A)
    
    def fH2dQ2(self, Q2):
        """Construire une fonction H2 à dériver par rapport au Q2
        Entrée :
            Rien
        Sortie : 
            Fonction H2 à dériver par rapport au Q2"""
        Q1_t = self.x[0] + self.deltaT*(-self.g*self.A*(self.x[1]-self.H_in)/self.x[3]
                                        - self.coef_frot(self.x[0])*self.x[0]**2/2/self.D/self.A)
        H2_t = self.x[1] + self.deltaT*(-self.b**2/(self.g*self.A*self.x[3])*(Q2 - self.x[0]
                                                                              + self.x[4]*np.sqrt(np.abs(self.x[1]))))
        Q2_t = Q2 + self.deltaT*(-self.g*self.A*(self.H_out-self.x[1])/(self.L-self.x[3])
                                 - self.coef_frot(Q2)*Q2**2/(2*self.D*self.A))
        return self.x[1] + self.deltaT/2*(-self.b**2/(self.g*self.A*self.x[3])*(Q2 - self.x[0]
                                                                                + self.x[4]*np.sqrt(np.abs(self.x[1])))
                                          -self.b**2/(self.g*self.A*self.x[3])*(Q2_t - Q1_t
                                                                                + self.x[4]*np.sqrt(np.abs(H2_t))))
    
    def fQ2dQ2(self, Q2):
        """Construire une fonction Q2 à dériver par rapport au Q2
        Entrée :
            Rien
        Sortie : 
            Fonction Q2 à dériver par rapport au Q2"""
        H2_t = self.x[1] + self.deltaT*(-self.b**2/(self.g*self.A*self.x[3])*(Q2 - self.x[0]
                                                                              + self.x[4]*np.sqrt(np.abs(self.x[1]))))
        Q2_t = Q2 + self.deltaT*(-self.g*self.A*(self.H_out-self.x[1])/(self.L-self.x[3])
                                 - self.coef_frot(Q2)*Q2**2/(2*self.D*self.A))
        return Q2 + self.deltaT/2*(-self.g*self.A*(self.H_out-self.x[1])/(self.L-self.x[3])
                                   - self.coef_frot(Q2)*Q2**2/2/self.D/self.A
                                   - self.g*self.A*(self.H_out-H2_t)/(self.L-self.x[3])
                                   - self.coef_frot(Q2_t)*Q2_t**2/2/self.D/self.A)
    def fQ1dzL(self, zL):
        """Consrtuire une fonction Q1 à dériver par rapport at zL
        Entrée : 
            Rien
        Sortie : 
            Fonction Q1 à dériver par rapport au zL"""
        Q1_t = self.x[0] + self.deltaT*(-self.g*self.A*(self.x[1]-self.H_in)/zL
                                        - self.coef_frot(self.x[0])*self.x[0]**2/2/self.D/self.A)
        H2_t = self.x[1] + self.deltaT*(-self.b**2/(self.g*self.A*zL)*(self.x[2] - self.x[0]
                                                                       + self.x[4]*np.sqrt(np.abs(self.x[1]))))
        return self.x[0] + self.deltaT/2*(-self.g*self.A*(self.x[1]-self.H_in)/zL
                                          - self.coef_frot(self.x[0])*self.x[0]**2/2/self.D/self.A
                                          - self.g*self.A*(H2_t-self.H_in)/zL
                                          - self.coef_frot(Q1_t)*Q1_t**2/2/self.D/self.A)
    
    def fH2dzL(self, zL):
        """Construire une fonction H2 à dériver par rapport au zL
        Entrée :
            Rien
        Sortie : 
            Fonction H2 à dériver par rapport au zL"""
        Q1_t = self.x[0] + self.deltaT*(-self.g*self.A*(self.x[1]-self.H_in)/zL
                                        - self.coef_frot(self.x[0])*self.x[0]**2/2/self.D/self.A)
        H2_t = self.x[1] + self.deltaT*(-self.b**2/(self.g*self.A*zL)*(self.x[2] - self.x[0]
                                                                       + self.x[4]*np.sqrt(np.abs(self.x[1]))))
        Q2_t = self.x[2] + self.deltaT*(-self.g*self.A*(self.H_out-self.x[1])/(self.L-zL)
                                        - self.coef_frot(self.x[2])*self.x[2]**2/(2*self.D*self.A))
        return self.x[1] + self.deltaT/2*(-self.b**2/(self.g*self.A*zL)*(self.x[2] - self.x[0]
                                                                         + self.x[4]*np.sqrt(np.abs(self.x[1])))
                                          -self.b**2/(self.g*self.A*zL)*(Q2_t - Q1_t
                                                                         + self.x[4]*np.sqrt(np.abs(H2_t))))
    
    def fQ2dzL(self, zL):
       """Construire une fonction Q2 à dériver par rapport au Q2
       Entrée :
           Rien
       Sortie : 
           Fonction Q2 à dériver par rapport au Q2"""
       H2_t = self.x[1] + self.deltaT*(-self.b**2/(self.g*self.A*zL)*(self.x[2] - self.x[0]
                                                                      + self.x[4]*np.sqrt(np.abs(self.x[1]))))
       Q2_t = self.x[2] + self.deltaT*(-self.g*self.A*(self.H_out-self.x[1])/(self.L-zL)
                                       - self.coef_frot(self.x[2])*self.x[2]**2/(2*self.D*self.A))
       return self.x[2] + self.deltaT/2*(-self.g*self.A*(self.H_out-self.x[1])/(self.L-zL)
                                         - self.coef_frot(self.x[2])*self.x[2]**2/2/self.D/self.A
                                         - self.g*self.A*(self.H_out-H2_t)/(self.L-zL)
                                         - self.coef_frot(Q2_t)*Q2_t**2/2/self.D/self.A)
   
    def fQ1dlamb(self, lamb):
        """Consrtuire une fonction Q1 à dériver par rapport au lambda
        Entrée : 
            Rien
        Sortie : 
            Fonction Q1 à dériver par rapport au lambda"""
        Q1_t = self.x[0] + self.deltaT*(-self.g*self.A*(self.x[1]-self.H_in)/self.x[3]
                                        - self.coef_frot(self.x[0])*self.x[0]**2/2/self.D/self.A)
        H2_t = self.x[1] + self.deltaT*(-self.b**2/(self.g*self.A*self.x[3])*(self.x[2] - self.x[0]
                                                                              + lamb*np.sqrt(np.abs(self.x[1]))))
        return self.x[0] + self.deltaT/2*(-self.g*self.A*(self.x[1]-self.H_in)/self.x[3]
                                          - self.coef_frot(self.x[0])*self.x[0]**2/2/self.D/self.A
                                          - self.g*self.A*(H2_t-self.H_in)/self.x[3]
                                          - self.coef_frot(Q1_t)*Q1_t**2/2/self.D/self.A)
    
    def fH2dlamb(self, lamb):
        """Construire une fonction H2 à dériver par rapport au lambda
        Entrée :
            Rien
        Sortie : 
            Fonction H2 à dériver par rapport au lamnda"""
        Q1_t = self.x[0] + self.deltaT*(-self.g*self.A*(self.x[1]-self.H_in)/self.x[3]
                                        - self.coef_frot(self.x[0])*self.x[0]**2/2/self.D/self.A)
        H2_t = self.x[1] + self.deltaT*(-self.b**2/(self.g*self.A*self.x[3])*(self.x[2] - self.x[0]
                                                                              + lamb*np.sqrt(np.abs(self.x[1]))))
        Q2_t = self.x[2] + self.deltaT*(-self.g*self.A*(self.H_out-self.x[1])/(self.L-self.x[3])
                                        - self.coef_frot(self.x[2])*self.x[2]**2/(2*self.D*self.A))
        return self.x[1] + self.deltaT/2*(-self.b**2/(self.g*self.A*self.x[3])*(self.x[2] - self.x[0]
                                                                                + lamb*np.sqrt(np.abs(self.x[1])))
                                          -self.b**2/(self.g*self.A*self.x[3])*(Q2_t - Q1_t
                                                                                + lamb*np.sqrt(np.abs(H2_t))))
    
    def fQ2dlamb(self, lamb):
       """Construire une fonction Q2 à dériver par rapport au lambda
       Entrée :
           Rien
       Sortie : 
           Fonction Q2 à dériver par rapport au lambda"""
       H2_t = self.x[1] + self.deltaT*(-self.b**2/(self.g*self.A*self.x[3])*(self.x[2] - self.x[0]
                                                                             + lamb*np.sqrt(np.abs(self.x[1]))))
       Q2_t = self.x[2] + self.deltaT*(-self.g*self.A*(self.H_out-self.x[1])/(self.L-self.x[3])
                                       - self.coef_frot(self.x[2])*self.x[2]**2/(2*self.D*self.A))
       return self.x[2] + self.deltaT/2*(-self.g*self.A*(self.H_out-self.x[1])/(self.L-self.x[3])
                                         - self.coef_frot(self.x[2])*self.x[2]**2/2/self.D/self.A
                                         - self.g*self.A*(self.H_out-H2_t)/(self.L-self.x[3])
                                         - self.coef_frot(Q2_t)*Q2_t**2/2/self.D/self.A)
    
    def matrice_F(self, v):
        F = np.eye(5)
        F[0, 0] = derivative(self.fQ1dQ1, v[0], dx=1e-6)
        F[1, 0] = derivative(self.fH2dQ1, v[0], dx=1e-6)
        F[2, 0] = derivative(self.fQ2dQ1, v[0], dx=1e-6)
        F[0, 1] = derivative(self.fQ1dH2, v[1], dx=1e-6)
        F[1, 1] = derivative(self.fH2dH2, v[1], dx=1e-6)
        F[2, 1] = derivative(self.fQ2dH2, v[1], dx=1e-6)
        F[0, 2] = derivative(self.fQ1dQ2, v[2], dx=1e-6)
        F[1, 2] = derivative(self.fH2dQ2, v[2], dx=1e-6)
        F[2, 2] = derivative(self.fQ2dQ2, v[2], dx=1e-6)
        F[0, 3] = derivative(self.fQ1dzL, v[3], dx=1e-6)
        F[1, 3] = derivative(self.fH2dzL, v[3], dx=1e-6)
        F[2, 3] = derivative(self.fQ2dzL, v[3], dx=1e-6)
        F[0, 4] = derivative(self.fQ1dlamb, v[4], dx=1e-6)
        F[1, 4] = derivative(self.fH2dlamb, v[4], dx=1e-6)
        F[2, 4] = derivative(self.fQ2dlamb, v[4], dx=1e-6)
        return F
