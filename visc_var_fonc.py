import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal as sc 

############# Conditions initiales ##############

def CI(parametre):
    N, Umoy, Bmoy, centre, eps = parametre[0], parametre[4], parametre[5], parametre[9], parametre[12] # centre sert a changer la couche ou l'on injecte l'energie
    V = np.zeros(N, dtype=complex)
    B = np.zeros(N, dtype=complex)
    sigma_k = 1  # largeur de la gaussienne en n indice du kn
    
    for i in range(N):
        enveloppe = np.exp(-(((i-centre)/eps)**2) / (2 * sigma_k**2))  # gaussienne en n
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

############################ viscosité variable ##########################

def var_visc(dt, constante, activer, nu, eta):
    if activer:
        nu = constante/dt
        eta = nu
        return nu, eta
    else:
        return nu, eta

############### Helicité cinétique et CI liées #####################  /!\ WORK IN PROGRESS (sert a rien pour l'instant)

""" Le but serait d'inverser l'expression de Hm et Hc pour avoir les U et B initiaux a 
mettre si on veut initialement Hc et Hb d'une certaine vaelur"""

def Hc(U, parametre):
    k = parametre[2]

    terme = k * (np.conj(U)**2 - U**2)
    Hc = 1j/2 * np.sum(terme, axis=1)
    return Hc

#################### Calculs des termes non linéaire et autre coeffs #######################

def signe(N): # pour avoir le (-1)**n
    return np.array([(-1)**(i+1) for i in range(N)]) 

def coeff(dt, pas_m1, pas_m2): # pour avoir les coeffs de la méthode AB 3
    terme = (dt/pas_m1) * (2*dt + 6*pas_m1 + 3*pas_m2)/(pas_m1 + pas_m2) + 6
    terme2 = -(dt/pas_m1 * (2*dt + 3* pas_m1 + 3*pas_m2)/pas_m2)
    terme3 = dt/pas_m2 * (2*dt + 3*pas_m1)/(pas_m1 + pas_m2)
    return terme, terme2, terme3

def NL(V, B, parametre):
    N, k, di, eps = parametre[0], parametre[2], parametre[6], parametre[12]

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
    
    beta =  (1 - 2**(2/eps))/(2**(2/eps) + 2**(3/eps) )
    zeta =  -(1 + 2**(-1/eps))/(2**(2/eps) + 2**(3/eps) )
    gamma =  1/(2**(2/eps) + 2**(1/eps) )
    
    
    NLV = 1j*k*np.conj(nonL1 + beta*nonL2 + zeta*nonL3) # calcul de la première equatiob (4) pour les N couches (a temps fixé)

    terme1 = 1j*k*gamma*np.conj( croiseNL1 + croiseNL2 + croiseNL3) # calcul de la deuxième equation (5) pour les N couches (a temps fixé)
    terme2 = signe(N)*di*1j*k**2*np.conj(Bn1*Bn2 + Bm1*Bn1*beta + Bm2*Bm1*zeta)
    
    NLB = terme1 + terme2

    return NLV, NLB

#################### Intégration ########################

# (méthode d'Euler pour calculer le terme au temps t1)
def integ(v0, b0, parametre, nb_sauvegarde, V_nul):
    T_max, k, nu, eta, constante, activer = parametre[1], parametre[2], parametre[7], parametre[8], parametre[14], parametre[15]

    U = [v0.copy()] # !! il faut meettre le .copy pour éviter des erreurs éventuelles de partage mémoire (dans les autres programmes j'ai oublié)
    B = [b0.copy()]
    l_T = [0]
    liste_dt = [] # juste des données en plus, bien a avoir (merci Benoit ^^)
    l_visc = [(nu, eta)]
    t = 0
    Vmain = v0.copy()
    Bmain = b0.copy()
    k4 = ((k*k)**2)

    compteur = 0
    dt_m1 = dt_m2 = 0
    fapV = fapB = 0
    dt_save = T_max / nb_sauvegarde # nb_sauvegarde est le nb de point total de sauvegarde qu'on veut
    prochaine_save = dt_save

    while t < T_max:
        dt = var(Vmain, Bmain, parametre)
        nu, eta = var_visc(dt, constante, activer, nu, eta)
        visqn = nu * k4
        visqe = eta * k4

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
            l_visc.append((nu, eta))
            while prochaine_save <= t:
                prochaine_save += dt_save

        if compteur % 500_000 == 0:
                print(f"  t = {t:.2f} / {T_max}  (dt = {dt:.2e},  pts sauvés = {len(l_T)}), nu = {nu}, eta = {eta} ")

    U_f  = np.array(U)
    B_f  = np.array(B)
    l_T = np.array(l_T)
    liste_dt = np.array(liste_dt)
    l_visc = np.array(l_visc)

    return U_f, B_f, l_T, liste_dt, l_visc

############### Run de la simulation ###################

def run_simu(parametre, nb_sauvegarde, V_nul):
    v0, b0 = CI(parametre)
    V, B, l_T, liste_dt, visc = integ(v0, b0, parametre, nb_sauvegarde, V_nul)
    return V, B, l_T, liste_dt, visc

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


def tps_cascade(V, B, parametre, l_T, delta):
    """
    Petite doc sur cette fonction, car elle fait des noeuds dans le cerveaux a cause des conditions. (mémo pour pas oublier)
    On crée un seuil d'nergie sous lequel on considère que la cascade a deja démarrée. On regarde ensuite l'indice correspondant a ce moment.

    chaque valeur de la liste en dessous correspond a l'indice d'une energie en dessous du seuil
    np.where(r)[0] extrait les indices où la valeur est True et on prend la première valeur de cette liste d'indice vérifiant la condition
    """
    
    k, di, centre = parametre[2], parametre[6], parametre[9]

    amplitude_0 = np.sqrt(np.abs(V[0][centre])**2 + np.abs(B[0][centre])**2)
    tau = 1.0 / (di*k[centre]**2*amplitude_0 + k[centre]*amplitude_0 + 1e-20)

    l_E, _,_ = invariant(V, B, parametre)
    E0 = l_E[0]
    seuil = (1.0 - delta) * E0

    en_dessous = np.where(l_E <= seuil)[0]
    if en_dessous.size > 0:
        i_casc = en_dessous[0]
    else:
        i_casc = len(l_T) - 1
        print(f"ATTENTION----------------------: seulement {100*(1 - l_E[-1]/E0):.2f}% de E0 a disparu. On prend donc (len(l_T) -1) ")

    return l_T[i_casc], tau, l_E


################ moyenne Energie apres cascade ################

def tps_corr(l_T, signal):
    s = signal - np.mean(signal) # on centre
    n = len(s)

    corr = sc.correlate(s, s, mode='full', method='fft') # module scipy qui calcule la corrélation entre s et s avec FFT, car numpy était trop lent sans fft (c'éétait conseillé sur la doc numpy)
    corr = corr[n-1 : 2*n-1] # on prend que les positifs car np.correlate donne les temps décalés positifs et négatif symétriquement et les négatifs sortent avant   
    
    if corr[0] <= 0: 
        return l_T[-1] - l_T[0]  # renvoie le temps total si la variance est négative car alors le signal n'est jamais décorrelé
    corr = corr / corr[0] # car pour un élément ck = somme(s_{n-k} s_n) si k = 0 on obtient la variance a 1/N pres. Alors si on divise par corr[0] on a une correlation de 1 pour le signal avec lui meme
    dessous = np.where(corr < 1/np.e)[0] # on regarde les indices des correlation en dessous de 37% (la classique de l'exponentielle avec 63 % disparue)

    if len(dessous) == 0:
        return l_T[-1] - l_T[0] # pareil que pour une corrélation toujours présente, on renvoie le temps total en cas de probleme
    return l_T[dessous[0]] - l_T[0]



def tps_decorr(l_T, tau_corr, t_min, t_max, M):
    indices_choisis = []
    temps_actuel = t_min
    saut_min = l_T[1] - l_T[0] # au cas ou tau_corr s'approcherai trop du temps de corrélation (car on veut avoir un temps plus loin)
    pas = max(tau_corr, saut_min)
    
    while temps_actuel <= t_max and len(indices_choisis) < M:
        j = np.searchsorted(l_T, temps_actuel)  
        j = min(j, len(l_T) - 1) # récupère l'indice le plus proche de l_T que l'on veut. (car on tombe pas forcément pile sur une valeur dans la liste de temps)
        indices_choisis.append(int(j))
        temps_actuel += pas

    return np.array(indices_choisis)



def moy_E(parametre, l_T, Ekn, l_E, t_c, M):
    t_max = parametre[1]
    t_final = l_T[-1]
    
    if t_final <= t_c: # on sait jamais, sécurité
        print(f"PROBLEME!! t_final est plus petit que t_c: t_final = {t_final} et t_c = {t_c}")

    condition = (l_T > t_c) & (l_T < t_final)
    if np.sum(condition) < 5: # ne pas oublier!! les booléens valent 0 et 1, donc en sommant on compte les Trues  
        print("PETIT problème ----> pas assez de points entre t_c et t_final pour calculer la corrélation")

    dissipation = - np.gradient(l_E[condition], l_T[condition])
    temps_correlation = tps_corr(l_T[condition], dissipation) # on calcule le temps de corrélation de la dissipation

    l_ind = tps_decorr(l_T, temps_correlation, t_c, t_max, M)

    if len(l_ind) < 3: 
        print("Encore un PETIT problème --> temps de corrélation surement trop grand donc pas assez d'indices")

    E_ref = l_E[l_ind]
    spectre_norm = Ekn[l_ind] / E_ref[:, None]  # (pas oublier): le [:, None] sert a créer une dimension suplémentaire car Ekn est de dim 2. https://numpy.org/doc/stable/user/basics.indexing.html  
                                                # divise chaque spectre (de N = 25 couches par exemple) par son énergie totale a ce moment, pour le normaliser

    moy = np.mean(spectre_norm, axis=0) * np.mean(E_ref) # on redonne le poid ici après la moyenne
    
    return moy

def liss_E(moyEk):
    D = len(moyEk)
    liss_E_k = np.zeros(D)
    for i in range (D):
        if (i == D-1 or i == 0 ):
            liss_E_k[i] = moyEk[i]
        else :
            liss_E_k[i] = (moyEk[i+1] * moyEk[i-1] * moyEk[i])**(1/3)
    return liss_E_k

################## moyennage temporel pour un meme régime + moyennage sur tout le régime ############

def moyennage_1run(simulations_same): 
    Eu = []
    Eb = []
    l_tau_tc = []
    liste_T = []

    for simulation in simulations_same:
        V, B, parametre, l_T = simulation
        M, delta = parametre[3], parametre[13]

        t_c, tau, l_E = tps_cascade(V, B, parametre, l_T, delta)

        EkV,EkB = E_kn(V, B, parametre)
        Ek_V = moy_E(parametre, l_T, EkV, l_E, t_c, M)
        Ek_B = moy_E(parametre, l_T, EkB, l_E, t_c, M)
        Eu.append(Ek_V)
        Eb.append(Ek_B)
        l_tau_tc.append((tau, t_c))
        liste_T.append(l_T)
    
    Eu = np.array(Eu) # contient l'energie moyenné temporellement pour chaque run d'un meme régime
    Eb = np.array(Eb)
    l_tau_tc = np.array(l_tau_tc)

    return Eu, Eb, l_tau_tc, l_E, liste_T


def moyennage_regime(Eu, Eb):
    Eu_final = np.mean(Eu, axis = 0)
    Eb_final = np.mean(Eb, axis = 0)
    return Eu_final, Eb_final

############### calcul de dissipation ###############

def dissipation(Eu, Eb, parametre, nu_t, eta_t):
    """
    Eu, Eb     : taille N, spectre complexe sauvegardé après moyenne temporelle et sur les runs
    nu_t,eta_t: soit un scalaire (viscosite constante),
                soit un tableau de taille P (viscosite variable, alignee sur les temps sauves)
    retourne  : Du(t), Db(t)  de taille P
    """
    k  = parametre[2]
    k4 = (k*k)**2

    nu_t  = np.atleast_1d(nu_t)
    eta_t = np.atleast_1d(eta_t)
    if np.size(nu_t)  == 1: nu_t  = np.full(np.shape(Eu), nu_t[0])
    if np.size(eta_t) == 1: eta_t = np.full(np.shape(Eb), eta_t[0])

    Du = nu_t  * np.sum(k4 * 2*Eu)   # somme sur les couches (axis=1 en général mais la y'a plus le temps donc c'est de base l'axis 0)
    Db = eta_t * np.sum(k4 * 2*Eb)
    return Du, Db

############## Affichage #############

def auto_label(cas, para):
    extrait = {
        "di": f"$d_i={para["di"]}$", 
        "nu": f"$\\nu={para["nu"]}$",
        "eta": f"$\\eta={para["eta"]}$",
        "i": f"$centre={para["i"]}$",
        "N": f"$N={para["N"]}$",
        "M": f"$M={para["M"]}$",
        "eps": f"$\\epsilon={para["eps"]}$",
        "constante": f"$c={para["constante"]}$"
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


def show_Inv(inv_tot, labels):
    fig, axs = plt.subplots(3) 
    
    for regime, label in zip(inv_tot, labels):
        i = 0
        label_reg = label[i]
        preimere_courbe = True
        
        for inv in regime:
            l_E, l_Hm, l_Hh, l_T = inv

            if preimere_courbe:
                plot1 = axs[0].plot(l_T, l_E, label=label_reg, markersize=3)
                color = plot1[0].get_color()
                axs[1].plot(l_T, l_Hm, color=color, label=label_reg, markersize=3)
                axs[2].plot(l_T, l_Hh, color=color, label=label_reg, markersize=3)
            else:
                axs[0].plot(l_T, l_E, color=color, markersize=3)
                axs[1].plot(l_T, l_Hm, color=color, markersize=3)
                axs[2].plot(l_T, l_Hh, color=color, markersize=3)
            preimere_courbe = False
            i+=1

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

def show_dt(liste, labels):
    plt.figure()
    for (dt, temps), label in zip(liste, labels):
        plt.plot(temps, dt, label=label, markersize=4)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("t")
        plt.ylabel("pas d'intégration dt")
        plt.title("Evolution du pas d'intégration au cours du temps")

def show_EVB(simulations, labels, pentes, kdi, l_tau_tc):
    plt.figure()

    liste_Eu = []
    liste_Eb = []
    
    for (Eu, Eb, parametre), label, pente, k_di, tau_tc in zip(simulations, labels, pentes, kdi, l_tau_tc):
        k, nom = parametre[2], parametre[11]

        tau_tc = np.mean(tau_tc, axis=0)
        tau = tau_tc[0]
        t_c = tau_tc[1]

        if nom in ["HMHD_s", "HMHD_b"]:
            Eu = liss_E(Eu)
            liste_Eu.append(Eu)
            yV = Eu*k**(5/3)
            
            Eb = liss_E(Eb)
            liste_Eb.append(Eb)
            yB = Eb*k**(11/3)

            plt.plot(k, yB, label=label, markersize=4)
            plt.plot(k, [np.mean(yB[15:20]) for i in range(len(k))], '--')
            plt.plot(k, yV, label=label, markersize=4)
            plt.plot(k, [np.mean(yV[7:15]) for i in range(len(k))], '--')
            print(f"cascade a t={t_c:.1f} s en moyenne")

            Ek_fit, condition_fit, pente_fit = auto_fit(k, yB, 0.5, 200)
            plt.plot(k[condition_fit], Ek_fit, '--', label=f" pente {pente_fit}")
            plt.axvline(x=k_di,color='gray',linestyle='--', label = "kdi = 1")

            Ek_fit, condition_fit, pente_fit = auto_fit(k, yB, 700, 11000)
            plt.plot(k[condition_fit], Ek_fit, '--', label=f" pente {pente_fit}")

            package = (Eu, Eb, k)
        
        else:
            E = Eu + Eb
            E = liss_E(E)
            liste_Eu.append(E) # J'utilise liste_Eu mais ça contient la somme de Ev et Eb, c'est juste pour la praticité du return
            y = E*k**(-pente)
            
            plt.plot(k, y, label=label, markersize=4)
            plt.plot(k, [np.mean(y[7:15]) for i in range(len(k))], '--')
            print(f"cascade a t={t_c:.1f} s en moyenne, donc {t_c/tau:.2f} tau")
            print(f"tau:{tau} s en moyenne")

            Ek_fit, condition_fit, pente_fit = auto_fit(k, y, 1, 500)
            plt.plot(k[condition_fit], Ek_fit, '--', label=f" pente {pente_fit}")

            package = 0
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("nombre d'onde k (log)")
    plt.ylabel("Spectre d'energie avec compensation pentes respectives (log)")
    plt.title("Comparaison des spectres d'énergie entre régimes")
    plt.ylim(10**-25, 10**4)
    plt.legend(fontsize=8)
    
    return package, np.array(liste_Eu), np.array(liste_Eb)


def ratio_EB(package):
    Eu, Eb, k = package[0], package[1], package[2]
    plt.figure()
    
    y = Eu/Eb
    plt.plot(k, y)
    plt.plot(k, [1 for i in range(len(k))], '--')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("nombre d'onde k (log)")
    plt.ylabel("$E^u / E^b$")
    plt.title("Ratio entre l'énergie cinétique et magnétique")
    plt.ylim(10**-5, 10**4)


def show_diss(simulations_diss, labels):
    """
    simulations_diss : liste de tuples (l_Eu, l_Eb, liste_T, parametre, (nu_t, eta_t)) par regime.
    Pour la viscosite CONSTANTE : passer nu_t=parametre[7], eta_t=parametre[8].
    Pour la viscosite VARIABLE  : passer les tableaux nu(t), eta(t)
    liste_T: [l_T, l_T, l_T, ...] pour run 1, 2 , 3, ...
    """
    fig, axs = plt.subplots(2, sharex=True)
    
    for (l_Eu, l_Eb, liste_T, parametre, l_visc), label in zip(simulations_diss, labels):
        for Eu, Eb, l_T, visc in zip(l_Eu, l_Eb, liste_T, l_visc):

            nu_t, eta_t = zip(*visc)   # sert a découper une liste de tuple en n listes
            Du, Db = dissipation(Eu, Eb, parametre, nu_t, eta_t)
            
            p = axs[0].plot(l_T, Du, label=label)[0]
            axs[1].plot(l_T, Db, color=p.get_color(), label=label)

    axs[0].set_ylabel(r"$D^u(t)=\nu\sum k^4|V_n|^2$")
    axs[1].set_ylabel(r"$D^b(t)=\eta\sum k^4|B_n|^2$")
    axs[1].set_xlabel("t")
    
    for ax in axs:
        ax.set_yscale('log')
        ax.legend(fontsize=8)
   
    fig.suptitle("Taux de dissipation visqueuse au cours du temps")
    fig.tight_layout()

