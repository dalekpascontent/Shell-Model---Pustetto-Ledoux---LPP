def moy_E_k(V, B, parametre):
    N, P, k0, k, dt, Umoy, Bmoy, di, nu, eta = parametre

    Ekn = E_kn(V, B, parametre)
    E_t = np.sum(Ekn, axis=1)
    dE  = np.abs(np.gradient(E_t, dt))
    casc = np.argmax(dE > 1e-3*E_t[0]) #début de la cascade d'energie, on a quitté la zone laminaire

    debut = casc + int((P-casc)/10) # on avance un peu pour ne pas avoir la transition
    fin = P
    moy_E_k_zone = np.mean(Ekn[debut:fin], axis=0) # on moyenne sur cette zone temporelle

    return moy_E_k_zone

def liss_E(V,B,parametre):
    moy_E_k = moy_E_k(V,B,parametre)
    D = len(moy_E_k)
    liss_E_k = np.zeros(D)
    for i in range (D):
        if (i = D-1 or i = 0 ):
            liss_E_k[i] = moy_E_k[i]
        else :
            liss_E_k[i] = (moy_E_k[i+1] + moy_E_k[i-1] + moy_E_k[i])/3
            #liss_E_k[i] = (moy_E_k[i+1] * moy_E_k[i-1] * moy_E_k[i])**(1/3)
    return liss_E_k[i]
        
        
