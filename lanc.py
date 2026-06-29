import version_finale_fonctions as fm
import numpy as np
import matplotlib.pyplot as plt

############### Paramètres communs ###############

N = 25 # nombre de couches dans l'espace de fourier
k0 = 0.01
k = np.array([k0*(2**i) for i in range(N)])
time_moy = 0.05 #temps de moyennage temporel (ici 5% de T_max)
T_max    = 800  # temps physique total de simulation
nb_sauvegarde = 3000000  # c'est le nombre selon lequel  on sauvegarde des points temporels (tous les 200 points j'en sauvegarde 1, car sinon overflow)
lissage = False

############### Paramètres de simulation #############

regimes = {
    "HD": {"labels": "HydroDynamique di = 0, $\\nu = 10^{-13}$", "Umoyen": 1, "Bmoyen": 0, "di": 0, "nu": 1e-13, "eta": 1e-13},
    "EMHD": {"labels": "EMHD $di = 0.3, V= 0$, $\\eta = 10^{-14}$", "Umoyen": 0, "Bmoyen": 1, "di": 0.3, "nu": 1e-13, "eta": 1e-15},
}

"""
regimes = {
    "HD": {"labels": "HydroDynamique di = 0, $\\nu = 1^{-13}$", "Umoyen": 1, "Bmoyen": 0, "di": 0, "nu": 1e-13, "eta": 1e-13},
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
    
    if cas == "EMHD":
        V_nul = True
    else:
        V_nul = False

    U, B, l_T = fm.run_simu(parametre, nb_sauvegarde, V_nul)
    simulations.append((U, B, parametre, l_T))
    labels.append(para["labels"])
    print(f"Points et temps de simulation: {len(l_T)} , {l_T[-1]}")

fm.show_inv_multi(simulations, labels)
fm.show_E_k_multi(simulations, labels, lissage)


plt.show()