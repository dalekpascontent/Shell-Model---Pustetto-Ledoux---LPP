import fonctions_finales as ff
import numpy as np
import matplotlib.pyplot as plt

############### Paramètres communs ###############

k0 = 0.01
time_moy = 0.05 #temps de moyennage temporel (ici 5% de T_max)
nb_sauvegarde = 2000000  # c'est le nombre selon lequel  on sauvegarde des points temporels (tous les 200 points j'en sauvegarde 1, car sinon overflow)
lissage = True
mur = 0.3 # % de l'energie initiale permettant d'omologuer une couche comme étant "active" (donc avec encore assez d'energie pour ne pas etre négligeable)
############### Paramètres de simulation #############

regimes = {
    "HD": {"N": 25, "T_max": 300, "i": 2, "CFL": 0.05, "Umoyen": 1, "Bmoyen": 0, "di": 0, "nu": 1e-13, "eta": 1e-13},
    "EMHD": {"N": 25, "T_max": 500, "i": 5, "CFL": 0.4, "Umoyen": 0, "Bmoyen": 1, "di": 0.3, "nu": 1e-13, "eta": 1e-13},
}

"""
regimes = {
    "HD": {"labels": "HydroDynamique di = 0, $\\nu = 10^{-13}$", "Umoyen": 1, "Bmoyen": 0, "di": 0, "nu": 1e-13, "eta": 1e-13},
    "MHD": {"labels": "MHD di = 0", "Umoyen": 1, "Bmoyen": 1, "di": 0, "nu": 1e-13, "eta": 1e-13},
    "EMHD": {"labels": "EMHD $di = 0.3, V= 0$, $\\eta = 10^{-13}$", "Umoyen": 0, "Bmoyen": 1, "di": 0.3, "nu": 1e-13, "eta": 1e-13},
    "HMHD_s": {"labels": "HMHD di = 0.3, $\\eta = 1^{-13}$", "Umoyen": 1, "Bmoyen": 1, "di": 0.3, "nu": 1e-13, "eta": 1e-13},
    "HMHD_b": {"labels": "HMHD di = 1", "Umoyen": 1, "Bmoyen": 1, "di": 1, "nu": 1e-13, "eta": 1e-13}
}
"""

simulations = []
labels = []
liste = []
pentes_theoriques = {"HD": -5/3, "MHD": None, "EMHD": -7/3, "HMHD": -11/3}
pentes = []

############### RUn des simulations ##############

for cas, para in regimes.items():
    parametre = [para["N"], para["T_max"], time_moy, para["Umoyen"], para["Bmoyen"], para["di"], para["nu"], para["eta"], para["i"], para["CFL"]]
     #[nb de couches, temps max, k, % de moyennage, U départ, B départ, di, nu, eta, centre de gaussienne, coefficient CFL]
    k = np.array([k0*(2**i) for i in range(parametre[0])])
    parametre.insert(2, k)
    print(f"Simulation {cas}...")
    
    V_nul = (cas == "EMHD") # V_nul est une condition donc un booleen
    pentes.append(pentes_theoriques[cas])

    U, B, l_T, liste_dt = ff.run_simu(parametre, nb_sauvegarde, V_nul)
    simulations.append((U, B, parametre, l_T))
    labels.append(ff.auto_label(cas, para))
    liste.append((liste_dt, l_T))
    print(f"Points et temps de simulation: {len(l_T)} , {l_T[-1]}")

ff.show_Inv(simulations, labels)
ff.show_spectre(simulations, labels, pentes, mur)
ff.show_dt(liste, labels)

plt.show()