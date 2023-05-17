import numpy as np


fc = 96000
t_fin = 20 # en seconde
t = np.arange(0, t_fin+1, 1/fc)
print(t)
