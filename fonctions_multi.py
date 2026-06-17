
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import optimize



def CI(parametre):
    N, P, k0, k, dt, Umoy, Bmoy, di, nu, eta = parametre
    V = np.zeros(N, dtype=complex)
    B = np.zeros(N, dtype=complex)
    sigma_k = k0  # largeur de la gaussienne en k, centrée sur k0
    for i in range(N):
        enveloppe = np.exp(-(k[i] - 0.04)**2 / (4 * sigma_k**2))  # gaussienne en k
        phase = np.exp(1j * 2 * np.pi * np.random.rand())    # phase aléatoire
        V[i] = Umoy*enveloppe * phase   # pic à k0
        B[i] = Bmoy*enveloppe * phase   
    return V, B


############## Partie intégration ###############



# /!\ ATTENTION, les p qui vont suivre ne font pas reférence a la coquille n mais au p'ieme temps d'integration et le n correspond a la n'ieme coquille
# Chaque couple dans V et B contient 2 listes, qui correspondent a toutes les couches au temps tp


def NL(V, B, parametre):
    N, P, k0, k, dt, Umoy, Bmoy, di, nu, eta = parametre

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
   
    NLV = 1j*k*np.conj(Vn1*Vn2 - Bn1*Bn2 -1/4*(Vm1*Vn1 - Bm1*Bn1) -1/8*(Vm1*Vm2 -Bm1*Bm2)) # calcul de la première equatiob (4) pour les N couches (a temps fixé)

    signe = np.array([(-1)**(i+1) for i in range(N)]) # pour avoir le (-1)**n
    terme1 = 1j*k/6*np.conj( (Vn1*Bn2 - Bn1*Vn2) + (Vm1*Bn1 - Bm1*Vn1) + (Vm2*Bm1 - Bm2*Vm1)) # calcul de la deuxième equation (5) pour les N couches (a temps fixé)
    terme2 = signe*di*1j*k**2*np.conj(Bn1*Bn2 - Bm1*Bn1/4 - Bm2*Bm1/8)
    NLB = terme1 + terme2

    return NLV, NLB


#initialisation (méthode d'Euler pour calculer le terme au temps t1)
def integ(v0, b0, parametre, expV, expB):
    N, P, k0, k, dt, Umoy, Bmoy, di, nu, eta = parametre

    U = np.zeros((P, N), dtype=complex)
    B = np.zeros((P, N), dtype=complex)
    U[0] = v0
    B[0] = b0

    NLV, NLB = NL(v0, b0, parametre)

    Vmain = v0*expV + dt*NLV*expV #euler
    Bmain = b0*expB + dt*NLB*expB
    U[1] = Vmain
    B[1] = Bmain
    fapV, fapB = NLV, NLB

    for i in range(2, P):
        NLV, NLB = NL(Vmain, Bmain, parametre)
        V_2 = Vmain*expV + dt*(3/2*NLV*expV - 1/2*fapV*expV) 
        B_2 = Bmain*expB + dt*(3/2*NLB*expB - 1/2*fapB*expB)
        fapV, fapB = NLV, NLB  #(f(tp, ap), on l'avait calculé, on le sauvegarde pour le prochain calcul
        Vmain = V_2
        Bmain = B_2
        U[i] = Vmain   # on enregistre les valeurs au cours du temps
        B[i] = Bmain
    return U, B


################# Partie multi-régimes ################


def run_simu(parametre):
    N, P, k0, k, dt, Umoy, Bmoy, di, nu, eta = parametre
    
    expV = np.exp(-nu * (k**4) * dt)
    expB = np.exp(-eta * (k**4) * dt)
    v0, b0 = CI(parametre)
    V, B = integ(v0, b0, parametre, expV, expB)
    return V, B



################# Partie invariant ################


def invariant(V, B, parametre):
    N, P, k0, k, dt, Umoy, Bmoy, di, nu, eta = parametre

    l_E  = np.zeros(P)
    l_Hm = np.zeros(P)
    l_Hh = np.zeros(P)

    for i in range(P):
        for j in range(N):
            mod2_V = np.abs(V[i][j])**2   
            mod2_B = np.abs(B[i][j])**2  
            croise = (np.conj(V[i][j]) * B[i][j]).real  
            l_E[i]  += (mod2_V + mod2_B) / 2
            l_Hm[i] += ((-1)**(j+1)) * mod2_B / (2 * k[j])
            l_Hh[i] += ((-1)**(j+1) * di**2 * k[j] * mod2_V + di * 2 * croise) / 2
            
    d_E = np.gradient(l_E,dt)
    d_Hm = np.gradient(l_Hm,dt)
    d_Hh = np.gradient(l_Hh,dt)
    
    moy_dE = np.mean(d_E)
    moy_dHm = np.mean(d_Hm)
    moy_dHh = np.mean(d_Hh)
    
    l_T = [ dt*i for i in range(P)]
    return [l_E,l_Hm,l_Hh,d_E,d_Hm,d_Hh,moy_dE,moy_dHm,moy_dHh,l_T]


def E_kn(V, B, parametre):
    N, P, k0, k, dt, Umoy, Bmoy, di, nu, eta = parametre

    Ekn = np.zeros((P, N))
    for i in range(P):
        for j in range(N):
            mod2_V = np.abs(V[i][j])**2   
            mod2_B = np.abs(B[i][j])**2 
            temp = (mod2_V + mod2_B) / 2
            Ekn[i][j] = temp
    return Ekn

def show_inv (V : list, B : list, parametre ):
    
    l_E, l_Hm, l_Hh, d_E, d_Hm, d_Hh, moy_dE, moy_dHm, moy_dHh, l_T = invariant(V, B, parametre)
    
    print("Moyenne de la derivée de E :", moy_dE,"J/s")
    print("Moyenne de la derivée de Hm :", moy_dHm,"J.m/s")
    print("Moyenne de la derivée de Hh :", moy_dHh,"J.m/s")
    
    fig, axs = plt.subplots(3, sharex=True)
    axs[0].plot(l_T,d_E, 'x', label = " Dérivée de E (J/s) ")
    axs[1].plot(l_T,d_Hm, 'o', label = " Dérivée de Hm (J.m/s) ")
    axs[2].plot(l_T,d_Hh, '+', label = " Dérivée de Hh (J.m/S) ")
    plt.legend()

def show_inv_multi(simulations, labels):
    fig, axs = plt.subplots(3, sharex=True)
    for (V, B, parametre), label in zip(simulations, labels):
        l_E, l_Hm, l_Hh, d_E, d_Hm, d_Hh, moy_dE, moy_dHm, moy_dHh, l_T = invariant(V, B, parametre)
        axs[0].plot(l_T, d_E, label=label, markersize=3)
        axs[1].plot(l_T, d_Hm, label=label, markersize=3)
        axs[2].plot(l_T, d_Hh, label=label, markersize=3)
    axs[0].set_ylabel("dE/dt (J/s)")
    axs[1].set_ylabel("dHm/dt (J.m/s)")
    axs[2].set_ylabel("dHh/dt (J.m/s)")
    axs[2].set_xlabel("t (s)")
    for ax in axs:
        ax.legend(fontsize=8)
    fig.suptitle("Comparaison des invariants entre régimes")
    fig.tight_layout()


def moy_E_k(V,B, parametre):
    E_k = E_kn(V,B, parametre)
    moy_E_k = np.mean(E_k, axis=0)
    return moy_E_k

def show_E_k(V,B, parametre):
    k = parametre[3]

    plt.figure()
    plt.plot(k,moy_E_k(V,B, parametre), 'x')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Wavenumber")
    plt.ylabel("Spectral Energy (J.m/s)")

def show_E_k_multi(simulations, labels):
    plt.figure()
    for (V, B, parametre), label in zip(simulations, labels):
        k = parametre[3]
        plt.plot(k, moy_E_k(V, B, parametre), label=label, markersize=4)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Wavenumber")
    plt.ylabel("Spectral Energy (J.m/s)")
    plt.title("Comparaison des spectres d'énergie entre régimes")
    plt.legend(fontsize=8)

def fit_simple(X,Y, parametre):
    k = parametre[3]
    x = 10**4
    difference_array = np.absolute(X-x)
    index = difference_array.argmin()
    X_bis = X[:index]
    k_bis = k[:index]
    X_log = np.log10(X_bis)
    k_log = np.log10(k_bis)
    coef, cov  = np.polyfit (k_log,X_log, deg = 1, cov = True )
    
    k_fit = np.linespace(k_log[1],k_log[-1],10000)
    
    plt.figure()
    plt.plot(k_log,X_log, label = "Model")
    plt.plot(k_fit,coef[0]*k_fit + coef[1], label = "Linear reg")
    plt.legend()
