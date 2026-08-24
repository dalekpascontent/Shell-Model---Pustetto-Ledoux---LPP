
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import optimize



def CI(parametre):
    N, P, k0, k, dt, Umoy, Bmoy, di, nu, eta = parametre
    V = np.zeros(N, dtype=complex)
    B = np.zeros(N, dtype=complex)
    sigma_k = 1  # largeur de la gaussienne en n indice du kn
    
    for i in range(N):
        enveloppe = np.exp(-((i-2)**2) / (2 * sigma_k**2))  # gaussienne en n
        phase = np.exp(1j * 2 * np.pi * np.random.rand())    # phase aléatoire
        V[i] = Umoy*enveloppe * phase   # pic à k0
        B[i] = Bmoy*enveloppe * phase   
    
    E = np.sum(np.abs(V)**2 + np.abs(B)**2)/2 # sert a normaliser l'energie pour avoir le maximum a 1
    if E > 0: # juste pour pas diviser par 0 au cas ou
        V = V/np.sqrt(E)
        B = B/np.sqrt(E)

    return V, B


############### Variation du dt suivant la zone ####################

def var(V, B, parametre):  # méthode CFL
    k = parametre[3]
    nu = parametre[8]
    eta =  parametre[9]
    
    amplitude = (np.abs(V) + np.abs(B)) #revient au meme que NLn/Vn car NLn ~ k(V^2 + B^2) 
    max_ampli = np.max(amplitude) 
    ampli_NL = k* max_ampli
    condition = amplitude > 1e-10 * max_ampli
    visq = (nu + eta) * (k**4)

    NL_actif = np.where(condition, ampli_NL, 0) # fait un masque booleen sur la liste pour qu'on ne garde que les couches avec de l'energie (voir doc)
    visq_actif = np.where(condition, visq, 0)
    NL_max = np.max(NL_actif) # on prend bien toujours le pire terme (le plus grand)
    visq_max = np.max(visq_actif)

    delta_t = 0.25/(NL_max + visq_max)

    return delta_t


############## Partie intégration ###############



# /!\ ATTENTION, les p qui vont suivre ne font pas reférence a la coquille n mais au p'ieme temps d'integration et le n correspond a la n'ieme coquille
# Chaque couple dans V et B contient 2 listes, qui correspondent a toutes les couches au temps tp


def NL(V, B, parametre):
    N, T_max, k0, k, time, Umoy, Bmoy, di, nu, eta = parametre

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

    signe = np.array([(-1)**(i+1) for i in range(N)]) # pour avoir le (-1)**n
    terme1 = 1j*k/6*np.conj( croiseNL1 + croiseNL2 + croiseNL3) # calcul de la deuxième equation (5) pour les N couches (a temps fixé)
    terme2 = signe*di*1j*k**2*np.conj(Bn1*Bn2 - Bm1*Bn1/4 - Bm2*Bm1/8)
    
    NLB = terme1 + terme2

    return NLV, NLB

#################### Intégration ########################

# (méthode d'Euler pour calculer le terme au temps t1)
def integ(v0, b0, parametre):
    N, T_max, k0, k, time, Umoy, Bmoy, di, nu, eta = parametre

    U = [v0.copy()] # !! il faut meettre le .copy pour éviter des erreurs éventuelles de partage mémoire (dans les autres programmes j'ai oublié)
    B = [b0.copy()]
    l_T = [0]
    t = 0
    Vmain = v0.copy()
    Bmain = b0.copy()

    dt = var(U[0], B[0], parametre)

    NLV, NLB = NL(v0, b0, parametre)
    expV = np.exp(-nu * (k**4) * dt)
    expB = np.exp(-eta * (k**4) * dt)

    Vmain = Vmain * expV + dt * NLV * expV # Euler
    Bmain = Bmain * expB + dt * NLB * expB
    t += dt

    U.append(Vmain.copy())
    B.append(Bmain.copy())
    l_T.append(t)

    fapV, fapB = NLV, NLB
    dt_avant = dt
    exp_avantV = expV.copy() # pareil pour ne pas ecraser apres qu'on ait recalculé expV et expB
    exp_avantB = expB.copy()

    while t < T_max:
        dt = var(Vmain, Bmain, parametre)
        if t + dt > T_max: # cette ligne sert a combler le trou de la fin, car avec un pas de temps variable et une borne max le prochain dt pourrait dépasser
            dt = T_max - t
        if dt <=0: #petite sécurité
            print("PROBLEMEEEEEEEE -----------------")
            break

        NLV, NLB = NL(Vmain, Bmain, parametre)
        omega = dt/dt_avant
        expV = np.exp(-nu * (k**4) * dt)
        expB = np.exp(-eta * (k**4) * dt)

        V_2 = Vmain*expV + dt*((1+omega/2)*NLV*expV - (omega/2)*fapV*expV*exp_avantV) 
        B_2 = Bmain*expB + dt*((1+omega/2)*NLB*expB - (omega/2)*fapB*expB*exp_avantB)
        
        fapV, fapB = NLV, NLB  #(f(tp, ap), on l'avait calculé, on le sauvegarde pour le prochain calcul
        dt_avant = dt
        exp_avantV = expV.copy() 
        exp_avantB = expB.copy()
        Vmain = V_2
        Bmain = B_2
        t += dt
        U.append(Vmain.copy()) # on enregistre les valeurs au cours du temps
        B.append(Bmain.copy())
        l_T.append(t)

    U_f  = np.array(U)
    B_f  = np.array(B)
    l_T = np.array(l_T)

    return U_f, B_f, l_T


################# Partie multi-régimes ################


def run_simu(parametre):
    v0, b0 = CI(parametre)
    V, B, l_T = integ(v0, b0, parametre)
    return V, B, l_T

################# Partie invariant ################


def invariant(V, B, parametre):
    N, T_max, k0, k, time, Umoy, Bmoy, di, nu, eta = parametre

    mod2_V = np.abs(V)**2
    mod2_B = np.abs(B)**2
    croise1 = (np.conj(V) * B).real
    croise2 = (np.conj(B) * V).real
    signe = np.array([(-1)**(j+1) for j in range(N)])
    
    l_E  = np.sum((mod2_V + mod2_B) / 2, axis=1)   # (Somme sur l'axe 1 pour obtenir une taille P, l'axe 1 correspond anciennement a la somme sur j)
    l_Hm = np.sum(signe * mod2_B / (2 * k), axis=1)
    l_Hh = np.sum((signe * di**2 * k * mod2_V + di * (croise1 + croise2)) / 2, axis=1)

    return l_E,l_Hm,l_Hh


########### Partie Energie Ek ################

def E_kn(V, B, parametre): #sert a calculer l'energie totale. 
    k = parametre[3]

    mod2_V = np.abs(V)**2   
    mod2_B = np.abs(B)**2 
    Ekn = (mod2_V + mod2_B) / (2*k) # on divise par k pour avoir l'energie E(k) et pas E(kn) (on veut l'intégrale)

    return Ekn


def moy_E_k(V, B, parametre, l_T):
    time_moy = parametre[4]
    P = V.shape[0] # 1 lignes correspond a 1 coquille pour tous les temps différents. Donc on regarde la premiere ligne pour avoir le nombre de poitns temporel

    Ekn = E_kn(V, B, parametre)
    E_t = np.sum(Ekn, axis=1)
    dE = np.abs(np.gradient(E_t, l_T))  # gradient avec les vrais temps
    casc = np.argmax(dE < 1e-3 * E_t[0]) #début de la cascade d'energie, on a quitté la zone laminaire
    print(f"Pente {dE}, cascade = {casc}")

    if casc == 0: # pareil une sécurité pour les problemes particuliers
        casc = P // 10
        print("LA cascade est arrivée un peu trop vite o_O")
    
    debut = casc + int((P-casc)/10) # on avance un peu pour ne pas avoir la transition
    fin = debut + time_moy
    moy_E_k_zone = np.mean(Ekn[debut:fin], axis=0) # on moyenne sur cette zone temporelle

    return moy_E_k_zone

############# Partie d'affichage ##################

def show_inv_multi(simulations, labels):
    fig, axs = plt.subplots(3, sharex=True)
    for (V, B, parametre, l_T), label in zip(simulations, labels):
        l_E, l_Hm, l_Hh = invariant(V, B, parametre)
        axs[0].plot(l_T, l_E, label=label, markersize=3)
        axs[1].plot(l_T, l_Hm, label=label, markersize=3)
        axs[2].plot(l_T, l_Hh, label=label, markersize=3)
    axs[0].set_ylabel("E (J)")
    axs[1].set_ylabel("Hm (J.m)")
    axs[2].set_ylabel("Hh (J.m)")
    axs[2].set_xlabel("t (s)")
    for ax in axs:
        ax.legend(fontsize=8)
    fig.suptitle("Comparaison des invariants entre régimes")
    fig.tight_layout()


def show_E_k_multi(simulations, labels):
    plt.figure()
    for (V, B, parametre, l_T), label in zip(simulations, labels):
        k = parametre[3]
        Ek = moy_E_k(V, B, parametre, l_T)
        plt.plot(k, Ek, label=label, markersize=4)
        
        dEdk = np.abs(np.gradient(Ek, k)) 
        diss = np.argmax(dEdk < 1e-20)-3 #permet d'obtenir le moment on la courbe s'éffondre sans loi de puissance
        k_zone = k[2:diss] # permet d'isoler la zone de k pour la régression
        a, b = np.polyfit(np.log10(k_zone), np.log10(Ek[2:diss]), deg=1)
        E_fit = 10**(a * np.log10(k_zone) + b)
        plt.plot(k_zone, E_fit, '--', label=f"Fit (pente = {a:.2f})")

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Wavenumber")
    plt.ylabel("Spectral Energy (J.m/s)")
    plt.title("Comparaison des spectres d'énergie entre régimes")
    plt.ylim(10**-30, 10**2)
    plt.legend(fontsize=8)
