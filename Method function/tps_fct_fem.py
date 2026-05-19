"""
Solveur FEM pour la simulation thermique TPS
============================================
Méthode des éléments finis 2D avec éléments Q4 bilinéaires.

Points techniques :
  - Intégration de Gauss 2×2 pour les éléments intérieurs
  - Intégration 1D 2 points sur les faces de bord
  - Schéma d'Euler implicite en temps
  - Flux plasma avec schéma Crank-Nicolson
  - Condition de radiation non-linéaire résolue par Newton-Raphson
  - Sous-relaxation adaptative selon k pour stabiliser Newton

La formulation Newton pour la radiation est :
  K_rad = ∫ 4σε T³ ψᵢ ψⱼ dΓ   (matrice tangente)
  R_rad = ∫ σε (3T⁴ + T_stru⁴) ψᵢ dΓ   (résidu)
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

GAUSS_PTS = [-1/np.sqrt(3), 1/np.sqrt(3)]
GAUSS_WTS = [1.0, 1.0]


def position(X, Y, nx, ny):
    x = np.zeros((ny, nx))
    y = np.zeros((ny, nx))
    matrice_x = np.linspace(X[0], X[1], num=nx)
    matrice_y = np.linspace(Y[0], Y[1], num=ny)
    matrice_y = matrice_y[::-1]
    for m in range(ny):
        x[m, :] = matrice_x
    for n in range(nx):
        y[:, n] = matrice_y
    return x, y


def flux_plasma(t, t_entree, q_max):
    if 0 <= t <= t_entree:
        return q_max * np.sin(np.pi * t / t_entree)
    return 0.0


def shape_functions(xi, eta):
    N = np.array([
        (1-xi)*(1-eta)/4,
        (1+xi)*(1-eta)/4,
        (1+xi)*(1+eta)/4,
        (1-xi)*(1+eta)/4
    ])
    dN_dxi = np.array([
        -(1-eta)/4,
         (1-eta)/4,
         (1+eta)/4,
        -(1+eta)/4
    ])
    dN_deta = np.array([
        -(1-xi)/4,
        -(1+xi)/4,
         (1+xi)/4,
         (1-xi)/4
    ])
    return N, dN_dxi, dN_deta


def matrices_elem(k_therm, rho, cp, dx, dy):
    """Matrices de rigidité K_e et masse M_e pour un élément Q4, Gauss 2×2."""
    K_e = np.zeros((4, 4))
    M_e = np.zeros((4, 4))
    detJ = dx * dy / 4.0

    for xi, wi in zip(GAUSS_PTS, GAUSS_WTS):
        for eta, wj in zip(GAUSS_PTS, GAUSS_WTS):
            N, dN_dxi, dN_deta = shape_functions(xi, eta)
            dN_dx = dN_dxi  * (2.0 / dx)
            dN_dy = dN_deta * (2.0 / dy)
            K_e += k_therm * (np.outer(dN_dx, dN_dx)
                            + np.outer(dN_dy, dN_dy)) * detJ * wi * wj
            M_e += rho * cp * np.outer(N, N) * detJ * wi * wj

    return K_e, M_e


def vecteur_flux_plasma(q_mean, dx):
    """Flux plasma intégré sur un élément de bord avec Crank-Nicolson."""
    f = np.zeros(2)
    detJ_face = dx / 2.0
    for xi, wi in zip(GAUSS_PTS, GAUSS_WTS):
        N1D = np.array([(1-xi)/2, (1+xi)/2])
        f += N1D * q_mean * detJ_face * wi
    return f


def vecteur_radiation(T_nodes, sigma, epsilon, T_stru, dx):
    """
    Terme de radiation sur la face inférieure, linéarisé par Newton-Raphson.
    Retourne le vecteur résidu R_rad et la matrice tangente K_rad.
    """
    R_rad = np.zeros(2)
    K_rad = np.zeros((2, 2))
    detJ_face = dx / 2.0

    for xi, wi in zip(GAUSS_PTS, GAUSS_WTS):
        N1D = np.array([(1-xi)/2, (1+xi)/2])
        T_gauss = N1D @ T_nodes
        h_rad = 4.0 * sigma * epsilon * T_gauss**3
        K_rad += h_rad * np.outer(N1D, N1D) * detJ_face * wi
        r_rad = sigma * epsilon * (3.0 * T_gauss**4 + T_stru**4)
        R_rad += N1D * r_rad * detJ_face * wi

    return R_rad, K_rad


def assembler(T_iter, T_n, nx, ny, dx, dy, dt, prm, sigma, T_stru,
              q_mean, K_e, M_e):
    """
    Assemble le système global pour une itération Newton.

    T_iter est l'itéré courant (utilisé pour K_rad et R_rad).
    T_n est la température au pas de temps précédent (terme inertiel).

    Le système assemblé est :
      (M/dt + K + K_rad) T^{n+1} = M/dt T^n + f_plasma + R_rad
    """
    n_dof = nx * ny
    A = lil_matrix((n_dof, n_dof))
    b = np.zeros(n_dof)

    node = lambda i, j: i * nx + j
    Ae = M_e / dt + K_e

    # Éléments Q4 intérieurs
    for i in range(ny - 1):
        for j in range(nx - 1):
            dofs = [node(i+1, j), node(i+1, j+1),
                    node(i,   j+1), node(i,   j  )]
            fe = (M_e / dt) @ T_n[dofs]
            for a, dof_a in enumerate(dofs):
                b[dof_a] += fe[a]
                for bb, dof_b in enumerate(dofs):
                    A[dof_a, dof_b] += Ae[a, bb]

    # Face supérieure y=L : flux plasma CN
    for j in range(nx - 1):
        dofs_face = [node(0, j), node(0, j+1)]
        f_pl = vecteur_flux_plasma(q_mean, dx)
        for a, dof_a in enumerate(dofs_face):
            b[dof_a] += f_pl[a]

    # Face inférieure y=0 : radiation Newton-Raphson
    for j in range(nx - 1):
        dofs_face = [node(ny-1, j), node(ny-1, j+1)]
        T_face = T_iter[dofs_face]
        R_rad, K_rad_e = vecteur_radiation(T_face, sigma, prm.epsilon, T_stru, dx)
        for a, dof_a in enumerate(dofs_face):
            b[dof_a] += R_rad[a]
            for bb, dof_b in enumerate(dofs_face):
                A[dof_a, dof_b] += K_rad_e[a, bb]

    return A.tocsr(), b


def simulation_principale(X, Y, nx, ny, dt, t_final, temps_instantanes,
                          T_initiale, T_stru, t_entree, q_max, sigma, prm,
                          save_all=False, verbose=False):
    """
    Simulation FEM complète avec Newton-Raphson et sous-relaxation adaptative.

    La sous-relaxation est nécessaire pour les matériaux peu conducteurs
    (k < 1 W/m·K) où la radiation est très non-linéaire et Newton diverge
    sans amortissement :
      k < 0.5  →  ω = 0.2
      k < 1.0  →  ω = 0.4
      k ≥ 1.0  →  ω = 0.7
    """
    dx = (X[1] - X[0]) / (nx - 1)
    dy = (Y[1] - Y[0]) / (ny - 1)
    n_steps = int(t_final / dt)

    Position_x, Position_y = position(X, Y, nx, ny)
    T_vec = np.ones(nx * ny) * T_initiale

    temps           = np.zeros(n_steps + 1)
    T_max_evolution = np.zeros(n_steps + 1)
    T_face_interne  = np.zeros(n_steps + 1)
    T_instantanes        = []
    temps_instantanes_reels = []

    if save_all:
        T_field_complet = np.zeros((n_steps + 1, ny, nx))
        T_field_complet[0] = T_vec.reshape(ny, nx) - 273.15

    temps[0]           = 0.0
    T_face_interne[0]  = T_vec.reshape(ny, nx)[ny-1, nx//2] - 273.15
    T_max_evolution[0] = T_vec.max() - 273.15

    idx_instantanes = set([int(round(t / dt)) for t in temps_instantanes])

    if prm.k_therm < 0.5:
        omega = 0.2
    elif prm.k_therm < 1.0:
        omega = 0.4
    else:
        omega = 0.7

    K_e, M_e = matrices_elem(prm.k_therm, prm.rho, prm.cp, dx, dy)

    for step in range(1, n_steps + 1):
        t_actuel = step * dt
        q_t      = flux_plasma(t_actuel,      t_entree, q_max)
        q_t_old  = flux_plasma(t_actuel - dt, t_entree, q_max)
        q_mean   = (q_t + q_t_old) / 2.0

        T_n    = T_vec.copy()
        T_iter = T_n.copy()
        converged = False

        for niter in range(100):
            A, b = assembler(T_iter, T_n, nx, ny, dx, dy, dt,
                             prm, sigma, T_stru, q_mean, K_e, M_e)
            try:
                T_new_brut = spsolve(A, b)
            except Exception as e:
                print(f"Erreur solveur step {step}: {e}")
                T_new_brut = T_iter.copy()
                break

            T_new = (1 - omega) * T_iter + omega * T_new_brut
            diff  = np.max(np.abs(T_new - T_iter))
            T_iter = T_new.copy()

            if diff < 1e-6:
                converged = True
                break

        if not converged:
            print(f"  ⚠️ Non-convergence step {step} (diff={diff:.2e})")

        T_vec = T_iter
        T_2d  = T_vec.reshape(ny, nx)

        temps[step]           = t_actuel
        T_face_interne[step]  = T_2d[ny-1, nx//2] - 273.15
        T_max_evolution[step] = T_vec.max() - 273.15

        if save_all:
            T_field_complet[step] = T_2d - 273.15

        if step in idx_instantanes:
            T_instantanes.append(T_2d.copy() - 273.15)
            temps_instantanes_reels.append(t_actuel)

        if verbose and step % max(1, n_steps // 10) == 0:
            print(f"  {100*step/n_steps:.0f}%  t={t_actuel:.0f}s  "
                  f"T_max={T_vec.max()-273.15:.1f}°C")

    Resultats = {
        "Position_x"        : Position_x,
        "Position_y"        : Position_y,
        "temps"             : temps,
        "T_max_evolution"   : T_max_evolution,
        "T_face_interne"    : T_face_interne,
        "T_instantanes"     : T_instantanes,
        "temps_instantanes" : temps_instantanes_reels,
    }
    if save_all:
        Resultats["T_field_complet"] = T_field_complet

    return Resultats


def simulation_complete(nx, ny, dt, T_initiale, T_stru, t_entree,
                        sigma, t_final, L, q_max, prm):
    Res = simulation_principale(
        [0, L], [0, L], nx, ny, dt, t_final, [],
        T_initiale, T_stru, t_entree, q_max, sigma, prm,
        save_all=False, verbose=False
    )
    T_max_structure = np.max(Res["T_face_interne"])
    return T_max_structure, Res["temps"], Res["T_face_interne"]