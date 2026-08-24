import general as g
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize



T = 100 #Temps total de modélisation

N = 20
P = int(1e5)
k0 = 0.01
k = np.array([k0*(2**i) for i in range(N)])
dt = T / P
di = 0
nu = 1e-13
eta = 1e-13
parametre = [N, P, k0, k, dt, di, nu, eta] #[N, P, k0, k, dt, di, nu, eta]

Umoy = 1
Bmoy = 0

expV = np.exp(-nu*(k**4)*dt) #lissage pour V 
expB = np.exp(-eta*(k**4)*dt)

U = np.zeros((P, N), dtype=complex)
B = np.zeros((P, N), dtype=complex)


v0, b0 = g.CI(parametre, Umoy, Bmoy)
U1, B1 = g.integ(v0, b0, parametre, expV, expB)

g.show_E_k(U1, B1, parametre)
g.show_inv(U1, B1, parametre)

plt.show()
