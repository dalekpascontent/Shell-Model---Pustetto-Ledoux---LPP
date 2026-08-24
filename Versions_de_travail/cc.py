def invariant(V, B, parametre):
    N, P, k0, k, dt, di, nu, eta = parametre

    l_E  = np.zeros(P)
    l_Hm = np.zeros(P)
    l_Hh = np.zeros(P)

    for i in range(P):
        for j in range(N):
            mod2_V = np.abs(V[i][j])**2   
            mod2_B = np.abs(B[i][j])**2  
            croise = (np.conj(V[i][j]) * B[i][j]).real  
            if ( i => N ) :
                l_E[i] = (mod2_V + mod2_B)(k[i] -k[i-1]) / 2
            else :
                l_E[i]  += (mod2_V + mod2_B)(k[i+1] -k[i]) / 2
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
