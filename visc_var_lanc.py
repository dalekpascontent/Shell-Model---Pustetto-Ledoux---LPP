import visc_var_fonc as ff
import numpy as np
import matplotlib.pyplot as plt

############### Paramètres communs ###############

k0 = 0.01
nb_sauvegarde = 200000  # c'est le nombre selon lequel  on sauvegarde des points temporels (tous les 200 points j'en sauvegarde 1, car sinon overflow)
lissage = True
 
 
############### Paramètres de simulation #############

regimes = {
    "EMHD": {"N": 50, "T_max": 3000, "i": 6, "CFL": 0.4, "delta": 0.05, 
             "Umoyen": 0, "Bmoyen": 1, "di": 0.3, "nu": 1e-10, "eta": 1e-10, 
             "eps": 2, "M": 30, "constante": 1e-15, "visc_var": True},
}

"""
regimes = {
    "HD": {"N": 25, "T_max": 300, "i": 2, "CFL": 0.05, "delta": 0.05, "Umoyen": 1, "Bmoyen": 0, "di": 0, "nu": 1e-13, "eta": 1e-13, "eps": 1, "M": 30, "constante": 1e-18},
    "MHD": {"N": 25, "T_max": 400, "i": 3, "CFL": 0.1, "mur": 0.05, "Umoyen": 1, "Bmoyen": 1, "di": 0, "nu": 1e-12, "eta": 1e-12},
    "EMHD": {"N": 23, "T_max": 400, "i": 5, "CFL": 0.4, "delta": 0.05, "Umoyen": 0, "Bmoyen": 1, "di": 0.3, "nu": 1e-13, "eta": 1e-13, "eps": 1, "M": 30, "constante": 1e-17},
    "HMHD_s": {"N": 25, "T_max": 300, "i": 4, "CFL": 0.4, "delta": 0.05, "Umoyen": 1, "Bmoyen": 1, "di": 0.3, "nu": 1e-13, "eta": 1e-13, "eps": 1, "M": 30},
    "HMHD_b": {"N": 27, "T_max": 100, "i": 3, "CFL": 0.4, "mur": 0.05, "Umoyen": 1, "Bmoyen": 1, "di": 30, "nu": 1e-9, "eta": 1e-9}
}
"""

couleur = {"HD": "blue", "MHD": "yellow", "EMHD": "red", "HMHD_s": "green", "HMHD_b": "pink"}

simulations_tot = []
labels = []
pentes_theoriques = {"HD": -5/3, "MHD": -5/3, "EMHD": -7/3, "HMHD_s": -11/3, "HMHD_b": -11/3, "HMHD_s": -7/3, "HMHD_b": -11/3}
pentes = []
kdi = []
l_tau_tc = []
inv_total = []
simulations_visc = []

for cas, para in regimes.items():
    
    simulations_same = []
    inv = []
    l_visc = []
    parametre = [
        para["N"], 
        para["T_max"], 
        # Ici va venir k
        para["M"], # le nombre de spectres moyenné apres la cascade (on pourrait le mettre a l'exterieur, mais ça changerait l'indiçage, flemme)
        para["Umoyen"], 
        para["Bmoyen"], 
        para["di"], 
        para["nu"], 
        para["eta"],
        para["i"], 
        para["CFL"], 
        f"{cas}", 
        para["eps"],
        para["delta"], # (1 - delta) donne le % d'energie qu'on veut avoir dans le systeme pour detecter la cascade
        para["constante"],
        para["visc_var"]
        ]
    
    k = np.array([k0*(2**(i /parametre[11])) for i in range(parametre[0])])
    parametre.insert(2, k)

    V_nul = (cas == "EMHD")

    for j in range(3):
        print(f"Simulation {cas} numéro {j}...")

        U, B, l_T, liste_dt, visc = ff.run_simu(parametre, nb_sauvegarde, V_nul)
        l_E_1run, l_Hm, l_Hh = ff.invariant(U, B, parametre)
        
        inv.append((l_E_1run, l_Hm, l_Hh, l_T))
        labels.append(ff.auto_label(cas, para))
        simulations_same.append((U, B, parametre, l_T))
        l_visc.append(visc)
    print("run pour un régime finis")

    inv_total.append(inv)
    Eu, Eb, tau_tc, l_E, liste_T = ff.moyennage_1run(simulations_same)
    Eu_f, Eb_f = ff.moyennage_regime(Eu, Eb)
    print("moyennage fini")
    
    simulations_tot.append((Eu_f, Eb_f, parametre))
    simulations_visc.append((Eu, Eb, liste_T, parametre, l_visc))
    l_tau_tc.append(tau_tc) # liste de liste, l_tau_tc ici a la taille du nombre de régime, et chaque tau_tc est la liste des (tau, t_c) de chaque run
    
    pentes.append(pentes_theoriques[cas])
    
    condition = np.abs(parametre[6]*k -1)
    indice = np.argmin(condition)
    kdi.append(indice)


ff.show_Inv(inv_total, labels)
package, liste_Eu, liste_Eb = ff.show_EVB(simulations_tot, labels, pentes, kdi, l_tau_tc)
ff.show_diss(simulations_visc, labels)

if package == 0:
    None
else:
    ff.ratio_EB(package)

plt.show()