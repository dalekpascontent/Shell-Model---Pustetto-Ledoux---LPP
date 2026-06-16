def CI(V,B) : 
    for i in range(N):
        U = U_gaussian*np.sqrt(2*np.pi)*np.random.normal(scale = 8*k[i])/(2*k[i])
        B = B_gaussian*np.sqrt(2*np.pi)*np.random.normal(scale = 8*k[i])/(2*k[i])
        r_U = np.random.uniform()
        r_B = np.random.uniform()
        V[0][i] = U*r_U + (np.sqrt(1 -r_U**2)*U)*1j
        B[0][i] = B*r_B + (np.sqrt(1 -r_B**2)*B)*1j
