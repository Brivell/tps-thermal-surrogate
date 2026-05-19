"""
Fonctions pour la simulation TPS
GCH2545 - Projet de session
"""

import numpy as np
 
def position(X, Y, nx, ny):
    """
    Génère les matrices de position x et y
    """
    x = np.zeros((ny, nx))
    y = np.zeros((ny, nx))
    matrice_x = np.linspace(X[0], X[1], num=nx)
    matrice_y = np.linspace(Y[0], Y[1], num=ny)
    matrice_y = matrice_y[::-1]  # Inverser pour y décroissant
 
    for m in range(0, ny):
        x[m, :] = matrice_x
    for n in range(0, nx):
        y[:, n] = matrice_y
    return x, y
 
def flux_plasma(t, t_entree, q_max):
    """
    Calcule le flux de chaleur du plasma
    """
    # Flux sinusoidal durant l'entrée
    if 0 <= t <= t_entree:
        q = q_max * np.sin(np.pi * t / t_entree)
    else:
        q = 0.0
    return q
 
def euler_implicite_newton(T_current, X, Y, nx, ny, prm, sigma, T_stru, dt, t, t_entree, q_max, tol=1e-6, max_iter=50):
    """
    Résolution du système implicite avec linéarisation de Newton-Raphson
    pour le terme de rayonnement non-linéaire
    
    Schéma Euler implicite: (T^{n+1} - T^n)/dt = alpha * nabla^2(T^{n+1})
    Linéarisation rayonnement: T^4 ≈ h_rad * T
    """
    T_old = T_current.copy()
    
    # Calcul des pas d'espace
    dx = (X[1] - X[0]) / (nx - 1)
    dy = (Y[1] - Y[0]) / (ny - 1)
    k_map = lambda i, j: i * nx + j
    alpha = prm.k_therm / (prm.rho * prm.cp)
    q_t   = flux_plasma(t, t_entree, q_max)
    q_t_old = flux_plasma(t - dt, t_entree, q_max)
    q_mean  = (q_t + q_t_old) / 2.0
    converged = False
    n_iter = 0

    # Boucle Newton pour linéarisation du rayonnement
    for iteration in range(max_iter):
        n_iter += 1
        A = np.zeros((ny*nx, ny*nx))
        b = np.zeros(ny*nx)
        
        for i in range(ny):
            for j in range(nx):
                n = k_map(i, j)
                
                # Face supérieure (y = L): flux imposé
                if i == 0:
                    if j == 0:  # coin sup gauche
                        A[n, k_map(i, j)] = 3
                        A[n, k_map(i+1, j)] = -4
                        A[n, k_map(i+2, j)] = 1
                        b[n] = 2*dy*q_t/prm.k_therm
                    elif j == nx - 1:  # coin sup droit
                        A[n, k_map(i, j)] = 3
                        A[n, k_map(i+1, j)] = -4
                        A[n, k_map(i+2, j)] = 1
                        b[n] = 2*dy*q_t/prm.k_therm
                    else:  # bord supérieur
                        A[n, k_map(i, j)] = 3
                        A[n, k_map(i+1, j)] = -4
                        A[n, k_map(i+2, j)] = 1
                        q_t_old = flux_plasma(t - dt, t_entree, q_max)
                        q_mean  = (q_t + q_t_old) / 2.0
                        b[n] = 2 * dy * q_mean / prm.k_therm

                # Face inférieure (y = 0): rayonnement
                elif i == ny - 1:
                    T_old_surf = T_old[i, j]
                    # Coefficient de transfert radiatif linéarisé
                    h_rad = 4 * sigma * prm.epsilon * T_old_surf**3
                    
                    if j == 0:  # coin inf gauche
                        A[n, k_map(i, j)] = 3*prm.k_therm/(2*dy) + h_rad
                        A[n, k_map(i-1, j)] = -2*prm.k_therm/(dy)
                        A[n, k_map(i-2, j)] = prm.k_therm/(2*dy)
                        b[n] = h_rad*T_stru
                    elif j == nx - 1:  # coin inf droit
                        A[n, k_map(i, j)] = 3*prm.k_therm/(2*dy) + h_rad
                        A[n, k_map(i-1, j)] = -2*prm.k_therm/dy
                        A[n, k_map(i-2, j)] = prm.k_therm/(2*dy)
                        b[n] = h_rad*T_stru
                    else:  # bord inférieur
                        A[n, k_map(i, j)] = 3*prm.k_therm/(2*dy) + h_rad
                        A[n, k_map(i-1, j)] = -2*prm.k_therm/(dy)
                        A[n, k_map(i-2, j)] = prm.k_therm/(2*dy)
                        b[n] = h_rad*T_stru

                # Côtés latéraux: symétrie (dT/dx = 0)
                elif j == 0:  # bord gauche
                    A[n, k_map(i, j)] = -3
                    A[n, k_map(i, j+1)] = 4
                    A[n, k_map(i, j+2)] = -1
                    b[n] = 0
                elif j == nx - 1:  # bord droit
                    A[n, k_map(i, j)] = 3
                    A[n, k_map(i, j-1)] = -4
                    A[n, k_map(i, j-2)] = 1
                    b[n] = 0

                # Points intérieurs
                else:
                    A[n, k_map(i, j)] = 1 + 2*alpha*dt/(dx**2) + 2*alpha*dt/(dy**2)
                    A[n, k_map(i, j+1)] = -alpha*dt/(dx**2)
                    A[n, k_map(i, j-1)] = -alpha*dt/(dx**2)
                    A[n, k_map(i+1, j)] = -alpha*dt/(dy**2)
                    A[n, k_map(i-1, j)] = -alpha*dt/(dy**2)
                    b[n] = T_current[i, j]

        # Résolution du système linéaire
        try:
            T_vec = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            print(f"Erreur matrice singulière iter {iteration}")
            break
            
        T_new = T_vec.reshape((ny, nx))
        
        # Check convergence
        diff = np.max(np.abs(T_new - T_old))
        if diff < tol:
            converged = True
            break
        T_old = T_new.copy()
        
    if not converged and n_iter == max_iter:
        print(f"Attention: non-convergence après {max_iter} iterations")
        
    return T_new, converged, n_iter


def simulation_complete(nx, ny, dt, T_initiale, T_stru, t_entree, sigma, t_final, L, q_max, prm):
    """
    Simulation complète - retourne T_max pour tests de convergence
    """
    X = [0, L]
    Y = [0, L]
    n_steps = int(t_final / dt)
    
    # Init
    T = np.ones((ny, nx)) * T_initiale
    
    # Stockage
    temps = np.zeros(n_steps + 1)
    T_structure_evolution = np.zeros(n_steps + 1)
    temps[0] = 0.0
    T_structure_evolution[0] = T[ny-1, nx//2] - 273.15
    
    # Boucle temporelle
    for step in range(1, n_steps + 1):
        t = step * dt
        T_new, converged, n_iter = euler_implicite_newton(
            T, X, Y, nx, ny, prm, sigma, T_stru,
            dt, t, t_entree, q_max, tol=1e-6, max_iter=50
        )
        T = T_new
        temps[step] = t
        T_structure_evolution[step] = T[ny-1, nx//2] - 273.15
        
    T_max_structure = np.max(T_structure_evolution)
    
    return T_max_structure, temps, T_structure_evolution


def simulation_principale(X, Y, nx, ny, dt, t_final, temps_instantanes, 
                          T_initiale, T_stru, t_entree, q_max, sigma, prm, 
                          save_all=False, verbose=False):
    """
    Simulation complète avec sauvegarde des profils 2D.
    
    save_all=True  → sauvegarde T(x,y) à TOUS les instants
    save_all=False → sauvegarde seulement temps_instantanes
    """
    n_steps = int(t_final / dt)
    Position_x, Position_y = position(X, Y, nx, ny)
    
    T = np.ones((ny, nx)) * T_initiale
    
    # Stockage
    temps = np.zeros(n_steps + 1)
    T_max_evolution = np.zeros(n_steps + 1)
    T_face_interne = np.zeros(n_steps + 1)
    
    # Champ complet si save_all
    if save_all:
        T_field_complet = np.zeros((n_steps + 1, ny, nx))
        T_field_complet[0] = T - 273.15
    
    # Instantanés sélectifs
    T_instantanes = []
    temps_instantanes_reels = []
    
    # Convertir temps_instantanes en indices
    idx_instantanes = set([int(round(t / dt)) for t in temps_instantanes])
    
    temps[0] = 0.0
    T_face_interne[0] = T[ny-1, nx//2] - 273.15
    T_max_evolution[0] = np.max(T) - 273.15

    for step in range(1, n_steps + 1):
        t_actuel = step * dt
        T_new, converged, n_iter = euler_implicite_newton(
            T, X, Y, nx, ny, prm, sigma, T_stru,
            dt, t_actuel, t_entree, q_max, tol=1e-6, max_iter=50
        )
        T = T_new
        temps[step] = t_actuel
        T_face_interne[step] = T[ny-1, nx//2] - 273.15
        T_max_evolution[step] = np.max(T) - 273.15

        # Champ complet
        if save_all:
            T_field_complet[step] = T - 273.15

        # Instantanés sélectifs — comparaison par indice
        if step in idx_instantanes:
            T_instantanes.append(T.copy() - 273.15)
            temps_instantanes_reels.append(t_actuel)

        if verbose and step % max(1, n_steps // 10) == 0:
            print(f"  {100*step/n_steps:.0f}% (t={t_actuel:.0f}s)")

    Resultats = {
        "Position_x"      : Position_x,
        "Position_y"      : Position_y,
        "temps"           : temps,
        "T_max_evolution" : T_max_evolution,
        "T_face_interne"  : T_face_interne,
        "T_instantanes"   : T_instantanes,
        "temps_instantanes": temps_instantanes_reels,
    }
    
    # Ajouter champ complet si demandé
    if save_all:
        Resultats["T_field_complet"] = T_field_complet
        # shape: (n_steps+1, ny, nx) = (181, 21, 21)

    return Resultats