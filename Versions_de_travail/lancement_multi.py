import fonctions_multi as fm
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

############### Paramètres communs ###############

T = 100 #Temps total de modélisation
N = 20
P = int(1e5)
k0 = 0.01
k = np.array([k0*(2**i) for i in range(N)])
dt = T / P

############### Paramètres de simulation #############

regimes = {
    "HD": {"labels": "HydroDynamique di = 0, B=0", "Umoyen": 1, "Bmoyen": 0, "di": 0, "nu": 1e-13, "eta": 1e-13},
    "MHD": {"labels": "MHD di = 0", "Umoyen": 1, "Bmoyen": 1, "di": 0, "nu": 1e-13, "eta": 1e-13},
    "HMHD_s": {"labels": "HMHD di = 0.3", "Umoyen": 1, "Bmoyen": 1, "di": 0.3, "nu": 1e-13, "eta": 1e-13},
    "HMHD_b": {"labels": "HMHD di = 30", "Umoyen": 1, "Bmoyen": 1, "di": 1, "nu": 1e-13, "eta": 1e-13}
}


simulations = []
labels = []

############### RUn des simulations ##############

for cas, para in regimes.items():
    parametre = [N, P, k0, k, dt, para["Umoyen"], para["Bmoyen"], para["di"], para["nu"], para["eta"]] #[N, P, k0, k, dt, Umoy, Bmoy, di, nu, eta]

    U, B = fm.run_simu(parametre)
    simulations.append((U, B, parametre))
    labels.append(para["labels"])

U = np.zeros((P, N), dtype=complex)
B = np.zeros((P, N), dtype=complex)


U1, B1 = fm.run_simu(parametre)

fm.show_inv_multi(simulations, labels)
fm.show_E_k_multi(simulations, labels)

plt.show()
