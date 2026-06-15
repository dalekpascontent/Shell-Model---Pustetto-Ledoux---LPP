##PACKAGE

import numpy as np 
import matplotlib.pyplot as plt 


##PARAMETRES DU PROBLEME

# Parametres physiques : Unité U.S.I 
nu = 1
mu = 1 
k_0 = 1 
di = 1
#Parametres de modelisation 

N = 25 #Exposant du k_max
lmbd = 2 #facteur de puissance des k_n

P = 10**(5) #Indice de discretisation du temps
T = 100 #Temps total de modélisation
h = T/P

#Param de choix du modele 

HD = 1 # 0 pour enelever les effets cinétiques 
pB = 1 # 0 pour enlever effets magnetique
Hall = 1 # 0 pour enlever effet Hall

#Liste utile 

K = [k_0*(lmbd**i) for i in range(N)]



def lissage_lineaire ( V, B ) : 
    for i in range (P):
        for j in range(N):
            V[i][j][0] *= np.exp(-1*nu*(K[j]**2)*h)
            V[i][j][1] *= np.exp(-1*nu*(K[j]**2)*h)
            B[i][j][0] *= np.exp(-1*nu*(K[j]**2)*h)
            B[i][j][1] *= np.exp(-1*nu*(K[j]**2)*h)


def invariant (V :list ,B : list) : 
    l_E = np.array([0]*P)
    l_Hm = np.array([0]*P)
    l_Hh = np.array([0]*P)
    
    for i in range(P):
        for j in range (N):
            l_E[i] += ((V[i][j][0])**2 + (V[i][j][1])**2 + (B[i][j][0])**2 + (B[i][j][1])**2)/2
            l_Hm[i] += (((-1)**j)*((B[i][j][0])**2 + (B[i][j][1])**2)/(K[j]))/2
            l_Hh += ((((-1)**j)*(di**2)*(K[j])(V[i][j][0])**2 + (V[i][j][1])**2) + di*(V[i][j][1]*B[i][j][0] + V[i][j][0]*B[i][j][1] ))/2
    
    d_E = np.gradient(l_E,h)
    d_Hm = np.gradient(l_Hm,h)
    d_Hh = np.gradient(l_Hh,h)
    
    moy_dE = np.mean(d_E)
    moy_dHm = np.mean(d_Hm)
    moy_dHh = np.mean(d_Hh)
    
    l_T = [ h*i for i in range(P)]
    return [l_E,l_Hm,l_Hh,d_E,d_Hm,d_Hh,moy_dE,moy_dHm,moy_dHh,l_T]

def show_inv (V : list, B : list ):
    
    
    print("Moyenne de la derivée de E :", invariant(V,B)[6],"J/s")
    print("Moyenne de la derivée de Hm :", invariant(V,B)[7],"J.m/s")
    print("Moyenne de la derivée de Hh :", invariant(V,B)[8],"J.m/s")
    
    l_T = invariant(V,B)[9]
    d_E = invariant(V,B)[3]
    d_Hm = invariant(V,B)[4]
    d_Hh = invariant(V,B)[5]
    
    fig, axs = plt.subplots(3, sharex=True, sharey=True)
    axs[0].plot(l_T,d_E, label = " Dérivée de E (J/s) ")
    axs[1].plot(l_T,d_Hm, 'o', label = " Dérivée de Hm (J.m/s) ")
    axs[2].plot(l_T,d_Hh, '+', label = " Dérivée de Hh (J.m/S) ")
    plt.figure()
    plt.show()
    
