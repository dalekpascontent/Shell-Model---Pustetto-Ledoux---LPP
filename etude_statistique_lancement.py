import etude_stat_fonction as ff
import numpy as np
import matplotlib.pyplot as plt

############### Paramètres communs ###############

k0 = 0.01
time_moy = 0.05 #temps de moyennage temporel (ici 5% de T_max)
nb_sauvegarde = 2000000  # c'est le nombre selon lequel  on sauvegarde des points temporels (tous les 200 points j'en sauvegarde 1, car sinon overflow)
lissage = True
############### Paramètres de simulation #############

regimes = {
    "HD": {"N": 25, "T_max": 300, "i": 2, "CFL": 0.05, "mur": 0.05, "Umoyen": 1, "Bmoyen": 0, "di": 0, "nu": 1e-13, "eta": 1e-13},
}

"""
regimes = {
    "HD": {"labels": "HydroDynamique di = 0, $\\nu = 10^{-13}$", "Umoyen": 1, "Bmoyen": 0, "di": 0, "nu": 1e-13, "eta": 1e-13},
    "MHD": {"labels": "MHD di = 0", "Umoyen": 1, "Bmoyen": 1, "di": 0, "nu": 1e-13, "eta": 1e-13},
    "EMHD": {"N": 25, "T_max": 400, "i": 6, "CFL": 0.4, "Umoyen": 0, "Bmoyen": 1, "di": 0.3, "nu": 1e-13, "eta": 1e-13},
    "HMHD_s": {"labels": "HMHD di = 0.3, $\\eta = 1^{-13}$", "Umoyen": 1, "Bmoyen": 1, "di": 0.3, "nu": 1e-13, "eta": 1e-13},
    "HMHD_b": {"labels": "HMHD di = 1", "Umoyen": 1, "Bmoyen": 1, "di": 1, "nu": 1e-13, "eta": 1e-13}
}
"""

simulations = []
labels = []
liste = []
pentes_theoriques = {"HD": -5/3}
mure = [] # mur =  % de l'energie initiale permettant d'omologuer une couche comme étant "active" (donc avec encore assez d'energie pour ne pas etre négligeable)

############### RUn des simulations ##############

for j in range(25):
    for cas, para in regimes.items():
        parametre = [para["N"], para["T_max"], time_moy, para["Umoyen"], para["Bmoyen"], para["di"], para["nu"], para["eta"], para["i"], para["CFL"]]
        #[nb de couches, temps max, k, % de moyennage, U départ, B départ, di, nu, eta, centre de gaussienne, coefficient CFL]
        k = np.array([k0*(2**i) for i in range(parametre[0])])
        parametre.insert(2, k)
        print(f"Simulation {cas}...")
        
        V_nul = (cas == "EMHD") # V_nul est une condition donc un booleen
        mure.append(para["mur"])

        U, B, l_T, liste_dt = ff.run_simu(parametre, nb_sauvegarde, V_nul)
        simulations.append((U, B, parametre, l_T))
        liste.append((liste_dt, l_T))

ff.show_Inv(simulations)
liste_Ek = np.array(ff.show_spectre(simulations, pentes_theoriques["HD"], mure))
ff.etude_stat(liste_Ek, parametre, pentes_theoriques["HD"])

plt.show()