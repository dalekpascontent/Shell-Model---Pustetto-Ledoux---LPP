import dt as fm
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

############### Paramètres communs ###############

N = 25 # nombre de couches dans l'espace de fourier
k0 = 0.01
k = np.array([k0*(2**i) for i in range(N)])
time_moy = 50 #temps de moyennage temporel
T_max    = 130  # temps physique total de simulation

############### Paramètres de simulation #############

regimes = {
    "HD": {"labels": "HydroDynamique di = 0, $\\nu = 1^{-13}$", "Umoyen": 1, "Bmoyen": 0, "di": 0, "nu": 1e-13, "eta": 1e-13},
    "EMHD": {"labels": "EMHD $di = 0.3 V= 0$, $\\eta = 1^{-13}$", "Umoyen": 0, "Bmoyen": 1, "di": 0.3, "nu": 1e-13, "eta": 1e-10}
}

"""
regimes = {
    "HD": {"labels": "HydroDynamique di = 0, B=0", "Umoyen": 1, "Bmoyen": 0, "di": 0, "nu": 1e-13, "eta": 1e-13},
    "MHD": {"labels": "MHD di = 0", "Umoyen": 1, "Bmoyen": 1, "di": 0, "nu": 1e-13, "eta": 1e-13},
    "EMHD": {"labels": "EMHD $di = 0.3 V= 0$, $\\eta = 1^{-13}$", "Umoyen": 0, "Bmoyen": 1, "di": 0.3, "nu": 1e-13, "eta": 1e-10},
    "HMHD_s": {"labels": "HMHD di = 0.3, $\\eta = 1^{-13}$", "Umoyen": 1, "Bmoyen": 1, "di": 0.3, "nu": 1e-13, "eta": 1e-13},
    "HMHD_b": {"labels": "HMHD di = 1", "Umoyen": 1, "Bmoyen": 1, "di": 1, "nu": 1e-13, "eta": 1e-13}
}
"""

simulations = []
labels = []

############### RUn des simulations ##############

for cas, para in regimes.items():
    parametre = [N, T_max, k0, k, time_moy, para["Umoyen"], para["Bmoyen"], para["di"], para["nu"], para["eta"]] #[N, P, k0, k, time, Umoy, Bmoy, di, nu, eta]
    
    print(f"Simulation {cas}...")
    U, B, l_T = fm.run_simu(parametre)
    simulations.append((U, B, parametre, l_T))
    labels.append(para["labels"])
    print(f"Points et temps de simulation: {len(l_T)} , {l_T[-1]}")

fm.show_inv_multi(simulations, labels)
fm.show_E_k_multi(simulations, labels)


plt.show()