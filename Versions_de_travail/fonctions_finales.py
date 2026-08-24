import numpy as np 
import matplotlib.pyplot as plt 

############# Conditions initiales ##############

def CI(parametre):
    N, Umoy, Bmoy, centre = parametre[0], parametre[4], parametre[5], parametre[9] # centre sert a changer la couche ou l'on injecte l'energie
    V = np.zeros(N, dtype=complex)
    B = np.zeros(N, dtype=complex)
    sigma_k = 1  # largeur de la gaussienne en n indice du kn
    
    for i in range(N):
        enveloppe = np.exp(-((i-centre)**2) / (2 * sigma_k**2))  # gaussienne en n
        phaseV = np.exp(1j * 2 * np.pi * np.random.rand())    # phase aléatoire
        phaseB = np.exp(1j * 2 * np.pi * np.random.rand())    
        V[i] = Umoy*enveloppe * phaseV   # pic à k0
        B[i] = Bmoy*enveloppe * phaseB   
    
    E = np.sum(np.abs(V)**2 + np.abs(B)**2)/2 # sert a normaliser l'energie pour avoir le maximum a 1
    if E > 0: # juste pour pas diviser par 0 au cas ou
        V = V/np.sqrt(E)
        B = B/np.sqrt(E)

    return V, B

####################### pas de temps variable ########################

def var(V, B, parametre):  # méthode CFL (il n'y a pas le terme visqueux car k^4 mettait un dt trop petit + terme exact avec intégration exp)
    k, di, CFL = parametre[2], parametre[6], parametre[10]
    
    amplitude = (np.abs(V) + np.abs(B)) #revient au meme que NLn/Vn car NLn ~ k(V^2 + B^2) 
    max_ampli = np.max(amplitude) 
    condition = amplitude > 1e-10 * max_ampli
    terme = k*amplitude + di * (k*k) * np.abs(B)
    
    NL_max = np.max(np.where(condition, terme, 0)) # on prend bien toujours le pire terme (le plus grand), "where" fait un masque booleen sur la liste pour qu'on ne garde que les couches avec de l'energie (voir doc)
    delta_t = CFL/(NL_max + 1e-12) # le 1e-12 est pour ne pas diviser par 0 en sécurité

    return delta_t

#################### Calculs des termes non linéaire et autre coeffs #######################

def signe(N): # pour avoir le (-1)**n
    return np.array([(-1)**(i+1) for i in range(N)]) 

def coeff(dt, pas_m1, pas_m2): # pour avoir les coeffs de la méthode AB 3
    terme = (dt/pas_m1) * (2*dt + 6*pas_m1 + 3*pas_m2)/(pas_m1 + pas_m2) + 6
    terme2 = -(dt/pas_m1 * (2*dt + 3* pas_m1 + 3*pas_m2)/pas_m2)
    terme3 = dt/pas_m2 * (2*dt + 3*pas_m1)/(pas_m1 + pas_m2)
    return terme, terme2, terme3

def NL(V, B, parametre):
    N, k, di = parametre[0], parametre[2], parametre[6]

    V_grand = np.zeros(N+4, dtype = complex)
    B_grand = np.zeros(N+4, dtype = complex)
    V_grand[2: N+2] = V  # On a donc V_grand = [0, 0, V, 0, 0]
    B_grand[2: N+2] = B

    Vn1 = V_grand[3: N+3] #Vn+1
    Vn2 = V_grand[4: N+4] #Vn+2
    Vm1 = V_grand[1: N+1] #Vn-1
    Vm2 = V_grand[0: N] #Vn-2

    Bn1 = B_grand[3: N+3]
    Bn2 = B_grand[4: N+4]
    Bm1 = B_grand[1: N+1]
    Bm2 = B_grand[0: N]
    
    nonL1 = Vn1*Vn2 - Bn1*Bn2     #différentes contributions non linéaires qui vont servir à calculer le dt variable
    nonL2 = (Vm1*Vn1 - Bm1*Bn1)
    nonL3 = (Vm1*Vm2 -Bm1*Bm2)
    croiseNL1 = (Vn1*Bn2 - Bn1*Vn2)
    croiseNL2 = (Vm1*Bn1 - Bm1*Vn1)
    croiseNL3 = (Vm2*Bm1 - Bm2*Vm1)
    
    NLV = 1j*k*np.conj(nonL1 -1/4*nonL2 -1/8*nonL3) # calcul de la première equatiob (4) pour les N couches (a temps fixé)

    terme1 = 1j*k/6*np.conj( croiseNL1 + croiseNL2 + croiseNL3) # calcul de la deuxième equation (5) pour les N couches (a temps fixé)
    terme2 = signe(N)*di*1j*k**2*np.conj(Bn1*Bn2 - Bm1*Bn1/4 - Bm2*Bm1/8)
    
    NLB = terme1 + terme2

    return NLV, NLB

#################### Intégration ########################

# (méthode d'Euler pour calculer le terme au temps t1)
def integ(v0, b0, parametre, nb_sauvegarde, V_nul):
    T_max, k, nu, eta = parametre[1], parametre[2], parametre[7], parametre[8]

    U = [v0.copy()] # !! il faut meettre le .copy pour éviter des erreurs éventuelles de partage mémoire (dans les autres programmes j'ai oublié)
    B = [b0.copy()]
    l_T = [0]
    liste_dt = [] # juste des données en plus, bien a avoir (merci Benoit ^^)
    t = 0
    Vmain = v0.copy()
    Bmain = b0.copy()
    visqn = nu * ((k*k)**2)
    visqe = eta * ((k*k)**2)

    compteur = 0
    dt_m1 = dt_m2 = 0
    fapV = fapB = 0
    dt_save = T_max / nb_sauvegarde # nb_sauvegarde est le nb de point total de sauvegarde qu'on veut
    prochaine_save = dt_save

    while t < T_max:
        dt = var(Vmain, Bmain, parametre)
        if t + dt > T_max: # cette ligne sert a combler le trou de la fin, car avec un pas de temps variable et une borne max le prochain dt pourrait dépasser
            dt = T_max - t
        if dt <=0: #petite sécurité
            print("PROBLEMEEEEEEEE -----------------")
            break

        expV = np.exp(-visqn * dt)
        expB = np.exp(-visqe * dt)
        NLV, NLB = NL(Vmain, Bmain, parametre) # le terme le plus recent

        if V_nul:
            NLV *= 0 
        if compteur ==0: # Euler pour la première intégration
            V_2 = Vmain * expV + dt * NLV * expV 
            B_2 = Bmain * expB + dt * NLB * expB
            liste_dt.append(dt)
        elif compteur == 1:  # AB2 pour la deuxieme intégration
            omega = dt/dt_m1
            exp_m1_V = expV*np.exp(-visqn*dt_m1)
            exp_m1_B = expB*np.exp(-visqe*dt_m1)
            V_2 = Vmain*expV + dt*((1+omega/2)*NLV*expV - (omega/2)*fapV*exp_m1_V) 
            B_2 = Bmain*expB + dt*((1+omega/2)*NLB*expB - (omega/2)*fapB*exp_m1_B)
        else: # Le reste en AB3
            c1, c2, c3 = coeff(dt, dt_m1, dt_m2)
            exp_m1_V = expV*np.exp(-visqn*dt_m1)
            exp_m1_B = expB*np.exp(-visqe*dt_m1)
            exp_m2_V = exp_m1_V*np.exp(-visqn*dt_m2)
            exp_m2_B = exp_m1_B*np.exp(-visqe*dt_m2)
            V_2 = Vmain*expV + dt/6 * (c1*NLV*expV + c2*fapV*exp_m1_V + c3*fapV2*exp_m2_V)
            B_2 = Bmain*expB + dt/6 * (c1*NLB*expB + c2*fapB*exp_m1_B + c3*fapB2*exp_m2_B)

        if V_nul:
            V_2 *= 0

        fapV2, fapB2 = fapV, fapB #(f(tp-1, ap-1), on l'avait calculé, on le sauvegarde pour le prochain calcul
        fapV, fapB = NLV, NLB  #(f(tp, ap), on l'avait calculé, on le sauvegarde pour le prochain calcul
        dt_m2 = dt_m1 # "m1 signifie -1 (mémo si je m'en souviens plus)"
        dt_m1 = dt 
        Vmain = V_2
        Bmain = B_2
        t += dt
        compteur += 1

        if t >= prochaine_save:
            U.append(Vmain.copy())
            B.append(Bmain.copy())
            l_T.append(t)
            liste_dt.append(dt)
            while prochaine_save <= t:
                prochaine_save += dt_save

        if compteur % 500_000 == 0:
                print(f"  t = {t:.2f} / {T_max}  (dt = {dt:.2e},  pts sauvés = {len(l_T)})")

    U_f  = np.array(U)
    B_f  = np.array(B)
    l_T = np.array(l_T)
    liste_dt = np.array(liste_dt)

    return U_f, B_f, l_T, liste_dt

############### Run de la simulation ###################

def run_simu(parametre, nb_sauvegarde, V_nul):
    v0, b0 = CI(parametre)
    V, B, l_T, liste_dt = integ(v0, b0, parametre, nb_sauvegarde, V_nul)
    return V, B, l_T, liste_dt

################# Partie invariant ################


def invariant(V, B, parametre):
    N, k, di = parametre[0], parametre[2], parametre[6]

    mod2_V = np.abs(V)**2
    mod2_B = np.abs(B)**2
    croise1 = (np.conj(V) * B).real
    croise2 = (np.conj(B) * V).real
    s = signe(N)
    l_E  = np.sum((mod2_V + mod2_B) / 2, axis=1)   # (Somme sur l'axe 1 pour obtenir une taille P, l'axe 1 correspond anciennement a la somme sur j)
    l_Hm = np.sum(s * mod2_B / (2 * k), axis=1)
    l_Hh = np.sum((s * di**2 * k * mod2_V + di * (croise1 + croise2)) / 2, axis=1)

    return l_E,l_Hm,l_Hh

################### Calculs d'energie + détection cascade ######################

def E_kn(V, B, parametre): #sert a calculer l'energie 1/2 sum(Vn^2  +Bn^2). 
    k = parametre[2]

    mod2_V = 0.5 * np.abs(V)**2 / k  # divise par k car sinon on aurait pas la bonne dimension par rapport a Ekn qu'on veut
    mod2_B = 0.5 * np.abs(B)**2 / k

    return mod2_V, mod2_B

def tps_cascade(V, B, parametre, l_T, a):
    """
    Petite doc sur cette fonction, car elle fait des noeuds dans le cerveaux a cause des conditions. (mémo pour pas oublier)
    pic est bien un tableau. puis actif est unz matrice de booléens de la même taille que Ekn.
    Un élément vaut True si la couche n à l'instant t possède une énergie significative.
    On compare Chaque valeur de chaque couche à ce 'seuil' qui est a*pic pour déclarer la couche comme active

    chaque valeur de la liste plus_haute_c est le numéro de la couche avec la plus grande energie a un instant t donné.
    np.where(r)[0] extrait les indices où la valeur est True et on prend la couche avec le plus d'energie grace au max
    Si aucune couche n'est active, on met 0.
    """
    
    N, k, di, centre = parametre[0], parametre[2], parametre[6], parametre[9]
    
    amplitude_0 = np.sqrt(np.abs(V[0][0]**2 + B[0][0]**2)) # en soit c'est juste 1, car B0 et V0 = 1, mais au cas ou
    tau = 1 / (di*k[centre]**2*amplitude_0 + k[centre]*amplitude_0 + 1e-20) # temps caractéristique du systeme
    Ev, Eb = E_kn(V, B, parametre)
    Ekn = Ev + Eb

    objectif = N-2 # on veut regarder si la couche N-2, donc très petite echelle, a de l'energie de façon non négligeable, auquel cas la cascade a deja démarrée 
    pic = np.max(Ekn, axis=1, keepdims=True) # voir doc numpy, mais permet d'avoir une liste contenant la couche la plus energetique a chaque temps
    actif = Ekn > a * pic
    plus_haute_c = np.array([ np.max( np.where(instant)[0]) if instant.any() else 0 for instant in actif])
    condition = plus_haute_c >= objectif

    if np.any(condition):
        i_casc = np.argmax(condition)
    else:
        i_casc = len(l_T) - 1  # cascade non terminee en T_max
        print("cascade incomplete : la plus haute couche n'atteint pas la zone dissipative")
    return l_T[i_casc], tau

################ moyenne Energie apres cascade ################

def moy_E(parametre, l_T, Ekn, t_c):
    T_max, time_moy = parametre[1], parametre[3]

    debut = np.searchsorted(l_T, t_c)
    fin   = np.searchsorted(l_T, t_c + time_moy*T_max)
    moy_E_k_zone = np.mean(Ekn[debut:max(fin, debut+1)], axis=0)

    return moy_E_k_zone

def liss_E(moyEk):
    D = len(moyEk)
    liss_E_k = np.zeros(D)
    for i in range (D):
        if (i == D-1 or i == 0 ):
            liss_E_k[i] = moyEk[i]
        else :
            liss_E_k[i] = (moyEk[i+1] * moyEk[i-1] * moyEk[i])**(1/3)
    return liss_E_k

############## Affichage #############

def auto_label(cas, para):
    extrait = {
        "di": f"$d_i={para["di"]}$", 
        "nu": f"$\\nu={para["nu"]}$",
        "eta": f"$\\eta={para["eta"]}$",
        "i": f"$centre={para["i"]}$",
        "N": f"$N={para["N"]}$",
        "mur": f"$mur={para["mur"]}$"
    }
    label = [cas] + [extrait[clee] for clee in extrait if clee in para]
    return ", ".join(label)  # voir doc python pour le fonctionnement de .join (long a expliquer)

def auto_fit(k, Ek, k_min, k_max):
    
    condition = (k >= k_min) & (k <= k_max) # on ne peut pas juste faire un k_zone = k[k_min:k_max] malheureusement
    k_zone = k[condition]
    Ek_zone = Ek[condition]

    if len(k_zone) < 2:
        print("PROBLEME DE FIT, pas assez de point !!!!!!!!!!!!!!!!")
        return Ek, None, 0
    
    else:
        pente, origine = np.polyfit(np.log10(k_zone), np.log10(Ek_zone), 1)
        Ek_fit = 10**origine * k_zone**pente
        return Ek_fit, condition, pente


def show_Inv(simulations, labels):
    fig, axs = plt.subplots(3) 
    for (V, B, parametre, l_T), label in zip(simulations, labels):
        l_E, l_Hm, l_Hh = invariant(V, B, parametre)
        axs[0].plot(l_T, l_E, label=label, markersize=3)
        axs[1].plot(l_T, l_Hm, label=label, markersize=3)
        axs[2].plot(l_T, l_Hh, label=label, markersize=3)
        
    axs[0].set_ylabel("E")
    axs[1].set_ylabel("Hm")
    axs[2].set_ylabel("Hh")
    axs[2].set_xlabel("t")
    
    if np.any(l_Hm != 0):
        axs[1].set_xscale('log')
        axs[1].set_yscale('symlog') # symlog permet de ne pas avoir le probleme avec les Hm = 0 et négatifs si ça varie autour de 0
    
    for ax in axs:
        ax.legend(fontsize=8)
    fig.suptitle("Comparaison des invariants entre régimes (énergie, hélicité magnétique et hybride)")
    fig.tight_layout()

def show_spectre(simulations, labels, pentes, mure):
    plt.figure()
    for (V, B, parametre, l_T), label, pente, mur in zip(simulations, labels, pentes, mure):
        k = parametre[2]
        EkV, EkB = E_kn(V, B, parametre)
        Ekn = EkV + EkB
        t_c, tau = tps_cascade(V, B, parametre, l_T, mur)
        
        Ek = moy_E(parametre, l_T, Ekn, t_c)
        Ek = liss_E(Ek)
        y = Ek*k**(-pente)
        
        plt.plot(k, y, label=label, markersize=4)
        plt.plot(k, [np.mean(y[7:15]) for i in range(len(k))], '--')
        print(f"cascade a t={t_c:.1f} s  =  {t_c/tau:.2f} tau")
        print(f"tau:{tau} s")

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("nombre d'onde k (log)")
    plt.ylabel("Spectre d'energie avec compensation pentes respectives (log)")
    plt.title("Comparaison des spectres d'énergie entre régimes")
    plt.ylim(10**-25, 10**2)
    plt.legend(fontsize=8)

def show_dt(liste, labels):
    plt.figure()
    for (dt, temps), label in zip(liste, labels):
        plt.plot(temps, dt, label=label, markersize=4)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("t")
        plt.ylabel("pas d'intégration dt")
        plt.title("Evolution du pas d'intégration au cours du temps")

def show_EVB(simulations, labels, pentes, mure):
    plt.figure()

    for (V, B, parametre, l_T), label, pente, mur in zip(simulations, labels, pentes, mure):
        k, nom = parametre[2], parametre[11]
        EkV, EkB = E_kn(V, B, parametre)
        Ek = EkV + EkB

        if nom in ["HMHD_s", "HMHD_b"]:
            t_c = np.argmax(Ek <= 0.95)
            
            Ek_V = moy_E(parametre, l_T, EkV, t_c)
            Ek_V = liss_E(Ek_V)
            yV = Ek_V*k**(5/3)
            
            Ek_B = moy_E(parametre, l_T, EkB, t_c)
            Ek_B = liss_E(Ek_B)
            yB = Ek_B*k**(11/3)

            plt.plot(k, yB, label=label, markersize=4)
            plt.plot(k, [np.mean(yB[15:20]) for i in range(len(k))], '--')
            plt.plot(k, yV, label=label, markersize=4)
            plt.plot(k, [np.mean(yV[7:15]) for i in range(len(k))], '--')
            print(f"cascade a t={t_c:.1f} s ")

            Ek_fit, condition_fit, pente_fit = auto_fit(k, yB, 0.5, 200)
            plt.plot(k[condition_fit], Ek_fit, '--', label=f" pente {pente_fit}")

            package = (Ek_V, Ek_B, k)
        
        else:
            t_c, tau = tps_cascade(V, B, parametre, l_T, mur)

            E_k = moy_E(parametre, l_T, Ek, t_c)
            E_k = liss_E(E_k)
            y = E_k*k**(-pente)
            
            plt.plot(k, y, label=label, markersize=4)
            plt.plot(k, [np.mean(y[7:15]) for i in range(len(k))], '--')
            print(f"cascade a t={t_c:.1f} s  =  {t_c/tau:.2f} tau")
            print(f"tau:{tau} s")

            package = 0
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("nombre d'onde k (log)")
    plt.ylabel("Spectre d'energie avec compensation pentes respectives (log)")
    plt.title("Comparaison des spectres d'énergie entre régimes")
    plt.ylim(10**-25, 10**3)
    plt.legend(fontsize=8)
    
    return package

def ratio_EB(package):
    Ek_V, Ek_B, k = package[0], package[1], package[2]
    plt.figure()
    
    y = Ek_V/Ek_B
    plt.plot(k, y)
    plt.plot(k, [y[3] for i in range(len(k))], '--')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("nombre d'onde k (log)")
    plt.ylabel("$E^u / E^b$")
    plt.title("Ratio entre l'énergie cinétique et magnétique")
    plt.ylim(10**-5, 10**3)
            
