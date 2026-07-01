import numpy as np 
import matplotlib.pyplot as plt 


def CI(parametre):
    N, P, k0, k, dt, Umoy, di, nu, eta, eps = parametre
    V = np.zeros(N, dtype=complex)
    sigma_k = 1  # largeur de la gaussienne en n indice du kn
    
    for i in range(N):
        enveloppe = np.exp(-((i-2)**2) / (2 * sigma_k**2))  # gaussienne en n
        phaseV = np.exp(1j * 2 * np.pi * np.random.rand())    # phase aléatoire
        V[i] = Umoy*enveloppe * phaseV   # pic à k0
    
    E = np.sum(np.abs(V)**2)/2 # sert a normaliser l'energie pour avoir le maximum a 1
    if E > 0: # juste pour pas diviser par 0 au cas ou
        V = V/np.sqrt(E)

    return V


############### Variation du dt suivant la zone ####################

def var(V, parametre):  # méthode CFL (il n'y a pas le terme visqueux car k^4 mettait un dt trop petit + terme exact avec intégration exp)
    N, P, k0, k, dt, Umoy, di, nu, eta, eps = parametre
    
    amplitude = (np.abs(V)) #revient au meme que NLn/Vn car NLn ~ k(V^2 + B^2) 
    max_ampli = np.max(amplitude) 
    condition = amplitude > 1e-10 * max_ampli
    terme = k*amplitude
    
    NL_max = np.max(np.where(condition, terme, 0)) # on prend bien toujours le pire terme (le plus grand), "where" fait un masque booleen sur la liste pour qu'on ne garde que les couches avec de l'energie (voir doc)
    delta_t = 0.05/(NL_max)

    return delta_t


############## Partie calcul termes NL ###############


# /!\ ATTENTION, les p qui vont suivre ne font pas reférence a la coquille n mais au p'ieme temps d'integration et le n correspond a la n'ieme coquille
# Chaque couple dans V et B contient 2 listes, qui correspondent a toutes les couches au temps tp
def signe(N):
    return np.array([(-1)**(i+1) for i in range(N)]) # pour avoir le (-1)**n

def NL(V, parametre):
    N, T_max, k0, k, time, Umoy, di, nu, eta, eps = parametre

    V_grand = np.zeros(N+4, dtype = complex)
    V_grand[2: N+2] = V  # On a donc V_grand = [0, 0, V, 0, 0]

    Vn1 = V_grand[3: N+3] #Vn+1
    Vn2 = V_grand[4: N+4] #Vn+2
    Vm1 = V_grand[1: N+1] #Vn-11/8
    Vm2 = V_grand[0: N] #Vn-2
    
    nonL1 = Vn1*Vn2    #différentes contributions non linéaires qui vont servir à calculer le dt variable
    nonL2 = (Vm1*Vn1)
    nonL3 = (Vm1*Vm2)
    
    NLV = 1j*k*np.conj(nonL1 +((1 - 2**(2/eps))/(2**(2/eps)+2**(3/eps)))*nonL2 -((1 + 2**(-1/eps))/(2**(2/eps)+2**(3/eps)))*nonL3) # calcul de la première equatiob (4) pour les N couches (a temps fixé)


    return NLV
#################### Coeff d'Adam Bashforth 3d order ###############


def coeff(dt, pas_m1, pas_m2):
    terme = (dt/pas_m1) * (2*dt + 6*pas_m1 + 3*pas_m2)/(pas_m1 + pas_m2) + 6
    terme2 = -(dt/pas_m1 * (2*dt + 3* pas_m1 + 3*pas_m2)/pas_m2)
    terme3 = dt/pas_m2 * (2*dt + 3*pas_m1)/(pas_m1 + pas_m2)
    return terme, terme2, terme3

#################### Intégration ########################

# (méthode d'Euler pour calculer le terme au temps t1)
def integ(v0, parametre, nb_sauvegarde, V_nul):
    N, T_max, k0, k, time, Umoy, di, nu, eta , eps= parametre

    U = [v0.copy()] # !! il faut meettre le .copy pour éviter des erreurs éventuelles de partage mémoire (dans les autres programmes j'ai oublié)
    l_T = [0]
    t = 0
    Vmain = v0.copy()
    visqn = nu * ((k*k)**2)
    visqe = eta * ((k*k)**2)

    compteur = 0
    dt_m1 = dt_m2 = 0
    fapV = fapB = 0
    dt_save = T_max / nb_sauvegarde # nb_sauvegarde est le nb de point total de sauvegarde qu'on veut
    prochaine_save = dt_save

    while t < T_max:
        dt = var(Vmain, parametre)
        if t + dt > T_max: # cette ligne sert a combler le trou de la fin, car avec un pas de temps variable et une borne max le prochain dt pourrait dépasser
            dt = T_max - t
        if dt <=0: #petite sécurité
            print("PROBLEMEEEEEEEE -----------------")
            break

        expV = np.exp(-visqn * dt)
        NLV = NL(Vmain, parametre) # le terme le plus recent

        if V_nul:
            NLV *= 0 
        if compteur ==0: # Euler pour la première intégration
            V_2 = Vmain * expV + dt * NLV * expV 
        elif compteur == 1:  # AB2 pour la deuxieme intégration
            omega = dt/dt_m1
            exp_m1_V = expV*np.exp(-visqn*dt_m1)
            V_2 = Vmain*expV + dt*((1+omega/2)*NLV*expV - (omega/2)*fapV*expV*exp_m1_V) 
        else: # Le reste en AB3
            c1, c2, c3 = coeff(dt, dt_m1, dt_m2)
            exp_m1_V = expV*np.exp(-visqn*dt_m1)
            exp_m2_V = exp_m1_V*np.exp(-visqn*dt_m2)
            V_2 = Vmain*expV + dt/6 * (c1*NLV*expV + c2*fapV*exp_m1_V + c3*fapV2*exp_m2_V)

        if V_nul:
            V_2 *= 0

        fapV2 = fapV #(f(tp-1, ap-1), on l'avait calculé, on le sauvegarde pour le prochain calcul
        fapV = NLV #(f(tp, ap), on l'avait calculé, on le sauvegarde pour le prochain calcul
        dt_m2 = dt_m1 # "m1 signifie -1 (mémo si je m'en souviens plus)"
        dt_m1 = dt 
        Vmain = V_2
        t += dt
        compteur += 1

        if t >= prochaine_save:
            U.append(Vmain.copy())
            l_T.append(t)
            while prochaine_save <= t:
                prochaine_save += dt_save

        if compteur % 500_000 == 0:
                print(f"  t = {t:.2f} / {T_max}  (dt = {dt:.2e},  pts sauvés = {len(l_T)})")

    U_f  = np.array(U)
    l_T = np.array(l_T)

    return U_f, l_T


################# Partie multi-régimes ################


def run_simu(parametre, nb_sauvegarde, V_nul):
    v0 = CI(parametre)
    V, l_T = integ(v0, parametre, nb_sauvegarde, V_nul)
    return V, l_T

################# Partie invariant ################


def invariant(V, parametre):
    N, T_max, k0, k, time, Umoy, di, nu, eta, eps = parametre

    mod2_V = np.abs(V)**2
    l_E  = np.sum((mod2_V ) / 2, axis=1)   # (Somme sur l'axe 1 pour obtenir une taille P, l'axe 1 correspond anciennement a la somme sur j)


    return l_E


########### Partie Energie Ek ################

def E_kn(V, parametre): #sert a calculer l'energie totale. 
    k = parametre[3]

    mod2_V = np.abs(V)**2   
    Ekn = (mod2_V ) / (2*k) # on divise par k pour avoir l'energie E(k) et pas E(kn) (on veut l'intégrale)

    return Ekn

def E_kn_VB(V, B, parametre):
    k = parametre[3]
    return np.abs(V)**2/(2*k)


def moy_E_k(V, parametre, l_T):
    T_max = parametre[1]
    time_moy = parametre[4] 
    P = V.shape[0] # 1 lignes correspond a 1 coquille pour tous les temps différents. Donc on regarde la premiere ligne pour avoir le nombre de poitns temporel

    Ekn = E_kn(V, parametre)
    E_t = np.sum(Ekn, axis=1) # on somme sur les couches, pas le temps
    dE = np.abs(np.gradient(E_t, l_T))  # gradient avec les vrais temps
    condition = dE > 1e-3 * E_t[0]

    if not np.any(condition): # syntaxe trouvée sur internet pour vérfier une condition s'appliquant sur une liste comme un masque
        print("Cascade non trouvée")
    casc = np.argmax(condition) #début de la cascade d'energie, on a quitté la zone laminaire
    print(f"Pente {dE}, cascade = {casc}")
    if casc == 0: # pareil une sécurité pour les problemes particuliers
        casc = P // 2
        print("LA cascade est arrivée un peu trop vite o_O")
    
    debut = casc + int((P-casc)/10) # on avance un peu pour ne pas avoir la transition
    fin = debut + int(time_moy *T_max)
    moy_E_k_zone = np.mean(Ekn[debut:fin], axis=0) # on moyenne sur cette zone temporelle

    return moy_E_k_zone

def moy_Ek_VB(V, parametre, l_T):
    T_max = parametre[1]
    time_moy = parametre[4] 
    P = V.shape[0]

    EkV= E_kn_VB(V, parametre)
    Ekn = EkV
    E_t = np.sum(Ekn, axis=1) 
    dE = np.abs(np.gradient(E_t, l_T))  
    condition = dE > 1e-3 * E_t[0]
    if not np.any(condition): 
        print("Cascade non trouvée")
    casc = np.argmax(condition)
    print(f"Pente {dE}, cascade = {casc}")
    
    debut = casc + int((P-casc)/10) 
    fin = debut + int(time_moy *T_max)
    moy_E_k_zone_V = np.mean(EkV[debut:fin], axis=0)    

    return moy_E_k_zone_V

def liss_E(V,parametre,l_T):
    moyEk = moy_E_k(V,parametre,l_T)
    D = len(moyEk)
    liss_E_k = np.zeros(D)
    for i in range(D):
        if (i == D-1 or i == 0 ):
            liss_E_k[i] = moyEk[i]
        else :
            #liss_E_k[i] = (moyEk[i+1] + moyEk[i-1] + moyEk[i])/3
            liss_E_k[i] = (moyEk[i+1] * moyEk[i-1] * moyEk[i])**(1/3)
    return liss_E_k

############# Partie d'affichage ##################

def show_inv_multi(simulations, labels):
    plt.figure()
    for (V, parametre, l_T), label in zip(simulations, labels):
        l_E= invariant(V, parametre)
        plt.plot(l_T, l_E, label=label, markersize=3)
    plt.ylabel("E (J)")
    plt.xlabel("t (s)")
    plt.title("Comparaison des invariants entre régimes")


def show_E_k_multi(simulations, labels):
    plt.figure()
    for (V, parametre, l_T), label in zip(simulations, labels):
        k = parametre[3]
        #Ek = moy_E_k(V, parametre, l_T)
        Ek = liss_E(V,parametre,l_T)
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
    plt.ylim(10**-25, 10**2)
    plt.legend(fontsize=8)
    
def show_E_k_VB(simulations, labels):
    plt.figure()
    for (V,parametre, l_T), label in zip(simulations, labels):
        k = parametre[3]
        Ek = moy_E_k(V ,parametre, l_T)
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
    plt.ylim(10**-25, 10**2)
    plt.legend(fontsize=8)