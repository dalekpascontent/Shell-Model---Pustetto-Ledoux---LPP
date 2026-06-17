
def NL(V, B, parametre):
    N, P, k0, k, dt, di, nu, eta = parametre

    V_grand = np.zeros(N+4, dtype = complex)
    B_grand = np.zeros(N+4, dtype = complex)
    V_grand[2: N+2] = V  # On a donc V_grand = [0, 0, V, 0, 0]
    B_grand[2: N+2] = B

    V_grand[3: N+3] = V[N-1]**2:V[N-2]
    Vn1 = V_grand[3: N+3] #Vn+1
    V_grand[4: N+4] = Vn1*2:V[N-1]
    Vn2 = V_grand[4: N+4] #Vn+2
    
    V_grand[1: N+1] = V[0]**2/V[1]
    Vm1 = V_grand[1: N+1] #Vn-1
    V_grand[0: N] = Vm1**2/V[0]
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
