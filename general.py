
import numpy as np 
import matplotlib.pyplot as plt 


#Parametres de modelisation 

N = 23 #Exposant du k_max

P = int(1e5) #Indice de discretisation du temps
T = 100 #Temps total de modélisation
h = T/P
k0 = 0.04
di = 0
k = np.array([k0*(2**i) for i in range(N)])


def CI():
    V = np.zeros(N, dtype=complex)
    B = np.zeros(N, dtype=complex)
    sigma_k = k0  # largeur de la gaussienne en k, centrée sur k0
    for i in range(N):
        enveloppe = np.exp(-(k[i] - k0)**2 / (4 * sigma_k**2))  # gaussienne en k
        phase     = np.exp(1j * 2 * np.pi * np.random.rand())    # phase aléatoire
        V[i]      = enveloppe * phase   # amplitude O(1), pic à k0
    return V, B


############## Partie intégration ###############

v0, b0 = CI()
dt = T/P
nu = 1e-13
eta = 1e-13


U = np.zeros((P, N), dtype=complex)
B = np.zeros((P, N), dtype=complex)


# /!\ ATTENTION, les p qui vont suivre ne font pas reférence a la coquille n mais au p'ieme temps d'integration et le n correspond a la n'ieme coquille
# Chaque couple dans V et B contient 2 listes, qui correspondent a toutes les couches au temps tp


def NL(V, B):
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


expV = np.exp(-nu*(k**4)*dt) #lissage pour V 
expB = np.exp(-eta*(k**4)*dt)

#initialisation (méthode d'Euler pour calculer le terme au temps t1)
U[0] = v0
B[0] = b0

NLV, NLB = NL(v0, b0)

Vmain = v0*expV + dt*NLV*expV #euler
Bmain = b0*expB + dt*NLB*expB
U[1] = Vmain
B[1] = Bmain
fapV, fapB = NLV, NLB

for i in range(2, P):
    NLV, NLB = NL(Vmain, Bmain)
    V_2 = Vmain*expV + dt*(3/2*NLV*expV - 1/2*fapV*expV) 
    B_2 = Bmain*expB + dt*(3/2*NLB*expB - 1/2*fapB*expB)
    fapV, fapB = NLV, NLB  #(f(tp, ap), on l'avait calculé, on le sauvegarde pour le prochain calcul
    Vmain = V_2
    Bmain = B_2
    U[i] = Vmain   # on enregistre les valeurs au cours du temps
    B[i] = Bmain


################# Partie invariant ################


def invariant(V, B):
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
            
    d_E = np.gradient(l_E,h)
    d_Hm = np.gradient(l_Hm,h)
    d_Hh = np.gradient(l_Hh,h)
    
    moy_dE = np.mean(d_E)
    moy_dHm = np.mean(d_Hm)
    moy_dHh = np.mean(d_Hh)
    
    l_T = [ h*i for i in range(P)]
    return [l_E,l_Hm,l_Hh,d_E,d_Hm,d_Hh,moy_dE,moy_dHm,moy_dHh,l_T]


def E_kn(V, B):
    Ekn = np.zeros((P, N))
    for i in range(P):
        for j in range(N):
            mod2_V = np.abs(V[i][j])**2   
            mod2_B = np.abs(B[i][j])**2 
            temp = (mod2_V + mod2_B) / 2
            Ekn[i][j] = temp
    return Ekn

def show_inv (V : list, B : list ):
    
    l_E, l_Hm, l_Hh, d_E, d_Hm, d_Hh, moy_dE, moy_dHm, moy_dHh, l_T = invariant(V, B)
    
    print("Moyenne de la derivée de E :", moy_dE,"J/s")
    print("Moyenne de la derivée de Hm :", moy_dHm,"J.m/s")
    print("Moyenne de la derivée de Hh :", moy_dHh,"J.m/s")
    
    fig, axs = plt.subplots(3, sharex=True)
    axs[0].plot(l_T,d_E, label = " Dérivée de E (J/s) ")
    axs[1].plot(l_T,d_Hm, 'o', label = " Dérivée de Hm (J.m/s) ")
    axs[2].plot(l_T,d_Hh, '+', label = " Dérivée de Hh (J.m/S) ")
    plt.legend()
    plt.show()

def moy_E_k(V,B):
    E_k = E_kn(V,B)
    moy_E_k = np.mean(E_k, axis=0)
    return moy_E_k

def show_E_k(V,B):
    plt.figure()
    plt.plot(k,k**(5/3)*moy_E_k(V,B), 'x')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Wavenumber")
    plt.ylabel("Spectral Energy (J.m/s)")
    plt.show()

show_inv(U, B)
show_E_k(U, B)
