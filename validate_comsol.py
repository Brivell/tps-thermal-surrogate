"""
Validation des solveurs numériques contre COMSOL Multiphysics
=============================================================

On compare ici les deux solveurs maison (FDM et FEM) contre les données
de référence exportées depuis COMSOL. Les simulations COMSOL ont été
faites à t=600s (pic du flux plasma) pour deux balayages paramétriques :
  - variation de la conductivité thermique k
  - variation du flux plasma maximal q_max

En bas du fichier, on ajoute aussi un benchmark du surrogate GP
directement contre le FEM — les deux prédisent T_max global,
donc la comparaison est cohérente.

Fichiers requis :
  comsol_variation_k.txt
  comsol_variation_qmax.txt
  tps_fct_fdm.py
  tps_fct_fem.py
  train_surrogate_gp.py  (pour le benchmark GP)
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from multiprocessing import Pool, cpu_count
import pickle
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, "Method function"))

# Paramètres de simulation — identiques à COMSOL pour que la comparaison soit juste
SIGMA    = 5.67e-8
T_INIT   = 293.15
T_STRU   = 293.15
T_ENTREE = 1200.0
Q_MAX    = 50000.0   # utilisé pour le balayage k (q_max fixe)
L        = 0.1
NX = NY  = 21
DT       = 10.0
T_TARGET = 600.0     # instant COMSOL : pic du flux plasma


# =============================================================================
# Chargement des données COMSOL
# =============================================================================

def load_comsol_txt(filepath):
    # Le fichier COMSOL a quelques lignes d'en-tête qu'on détecte automatiquement
    skiprows = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                skiprows += 1
                continue
            try:
                float(line.split()[0])
                break
            except ValueError:
                skiprows += 1

    data   = np.loadtxt(filepath, skiprows=skiprows)
    params = data[:, 0]
    T_C    = data[:, 2] - 273.15   # COMSOL exporte en Kelvin, on convertit en °C
    print(f"  {filepath} — {len(params)} points")
    print(f"  Paramètre : [{params.min():.3f}, {params.max():.3f}]")
    print(f"  T range   : [{T_C.min():.1f}, {T_C.max():.1f}] °C")
    return params, T_C


# =============================================================================
# Fonctions de simulation (appelées en parallèle)
# =============================================================================

def run_fdm_one(args):
    k, L_val, q_max, t_target = args
    from tps_fct_fdm import simulation_principale

    class prm:
        rho=1800.0; cp=800.0; k_therm=k; epsilon=0.85

    Res = simulation_principale(
        [0, L_val], [0, L_val], NX, NY, DT,
        t_target, [], T_INIT, T_STRU, T_ENTREE, q_max,
        SIGMA, prm, save_all=False, verbose=False
    )
    return Res['T_max_evolution'][-1]


def run_fem_one(args):
    k, L_val, q_max, t_target = args
    from tps_fct_fem import simulation_principale

    class prm:
        rho=1800.0; cp=800.0; k_therm=k; epsilon=0.85

    Res = simulation_principale(
        [0, L_val], [0, L_val], NX, NY, DT,
        t_target, [], T_INIT, T_STRU, T_ENTREE, q_max,
        SIGMA, prm, save_all=False, verbose=False
    )
    return Res['T_max_evolution'][-1]


def run_parallel(func, args_list, n_cores=None):
    if n_cores is None:
        n_cores = cpu_count()
    print(f"  {n_cores} cores...")
    with Pool(processes=n_cores) as pool:
        results = pool.map(func, args_list)
    return np.array(results)


# =============================================================================
# Métriques de validation
# =============================================================================

def compute_metrics(T_ref, T_pred, name):
    r2  = r2_score(T_ref, T_pred)
    mae = mean_absolute_error(T_ref, T_pred)
    err = np.abs(T_pred - T_ref) / np.abs(T_ref) * 100
    print(f"\n  {name} vs COMSOL :")
    print(f"    R²         : {r2:.4f}")
    print(f"    MAE        : {mae:.2f} °C")
    print(f"    Erreur moy : {err.mean():.2f}%")
    print(f"    Erreur max : {err.max():.2f}%")
    print(f"    < 5%       : {(err<5).sum()}/{len(err)} points")
    return r2, mae, err


# =============================================================================
# Validation FDM + FEM vs COMSOL
# =============================================================================

def validate(param_vals, T_comsol, T_fdm, T_fem,
             param_name, param_unit, label, t_target):

    print(f"\n{'='*55}")
    print(f"Validation — {label}  (t={t_target}s)")
    print(f"{'='*55}")

    r2_fdm, mae_fdm, err_fdm = compute_metrics(T_comsol, T_fdm, "FDM")
    r2_fem, mae_fem, err_fem = compute_metrics(T_comsol, T_fem, "FEM")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f'Validation COMSOL vs FDM vs FEM — {label}\n'
        f't={t_target}s  ·  {param_name} ∈ '
        f'[{param_vals.min():.2f}, {param_vals.max():.2f}] {param_unit}',
        fontsize=13, fontweight='bold'
    )

    # Courbes T_max vs paramètre
    ax = axes[0]
    ax.plot(param_vals, T_comsol, 'k-o',  lw=2, ms=5, label='COMSOL (référence)')
    ax.plot(param_vals, T_fdm,    'b--s', lw=2, ms=5, label=f'FDM  (R²={r2_fdm:.3f})')
    ax.plot(param_vals, T_fem,    'r--^', lw=2, ms=5, label=f'FEM  (R²={r2_fem:.3f})')
    ax.set_xlabel(f'{param_name} ({param_unit})', fontsize=12)
    ax.set_ylabel('T_max (°C)', fontsize=12)
    ax.set_title(f'T_max vs {param_name}', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Diagramme de parité
    ax = axes[1]
    all_T = np.concatenate([T_comsol, T_fdm, T_fem])
    lims  = [all_T.min() - 50, all_T.max() + 50]
    ax.plot(lims, lims, 'k--', lw=1.5, label='Parfait (y=x)')
    ax.scatter(T_comsol, T_fdm, s=40, color='steelblue', alpha=0.8,
               label=f'FDM · MAE={mae_fdm:.0f}°C', zorder=5)
    ax.scatter(T_comsol, T_fem, s=40, color='tomato',    alpha=0.8,
               label=f'FEM · MAE={mae_fem:.0f}°C', zorder=5)
    ax.set_xlabel('T_max COMSOL (°C)', fontsize=12)
    ax.set_ylabel('T_max solveur (°C)', fontsize=12)
    ax.set_title('Parité', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Erreur relative par point
    ax = axes[2]
    x = np.arange(len(param_vals))
    w = 0.35
    ax.bar(x - w/2, err_fdm, w, color='steelblue', alpha=0.8,
           label=f'FDM  (moy {err_fdm.mean():.1f}%)')
    ax.bar(x + w/2, err_fem, w, color='tomato',    alpha=0.8,
           label=f'FEM  (moy {err_fem.mean():.1f}%)')
    ax.axhline(5, color='k', linestyle='--', lw=1.5, label='Seuil 5%')
    ax.set_xlabel(f'Indice ({param_name})', fontsize=12)
    ax.set_ylabel('Erreur relative (%)', fontsize=12)
    ax.set_title('Erreur relative FDM vs FEM', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fname = f'validate_comsol_{label}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"\n  Figure sauvegardée : {fname}")
    plt.show()

    return {
        'r2_fdm': r2_fdm, 'mae_fdm': mae_fdm, 'err_fdm': err_fdm,
        'r2_fem': r2_fem, 'mae_fem': mae_fem, 'err_fem': err_fem,
    }


# =============================================================================
# Graphique résumé FDM/FEM
# =============================================================================

def plot_summary(results_k, results_q):
    labels  = ['Variation k', 'Variation q_max']
    r2_fdm  = [results_k['r2_fdm'],  results_q['r2_fdm']]
    r2_fem  = [results_k['r2_fem'],  results_q['r2_fem']]
    mae_fdm = [results_k['mae_fdm'], results_q['mae_fdm']]
    mae_fem = [results_k['mae_fem'], results_q['mae_fem']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Résumé validation COMSOL vs FDM vs FEM',
                 fontsize=14, fontweight='bold')

    x = np.arange(len(labels))
    w = 0.35

    bars1 = ax1.bar(x - w/2, r2_fdm, w, color='steelblue', alpha=0.8, label='FDM')
    bars2 = ax1.bar(x + w/2, r2_fem, w, color='tomato',    alpha=0.8, label='FEM')
    ax1.axhline(0.99, color='k', linestyle='--', lw=2, label='Seuil 0.99')
    ax1.set_ylim(0.80, 1.01)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel('R²', fontsize=12)
    ax1.set_title('R² vs COMSOL', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(list(bars1)+list(bars2), r2_fdm+r2_fem):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    bars1 = ax2.bar(x - w/2, mae_fdm, w, color='steelblue', alpha=0.8, label='FDM')
    bars2 = ax2.bar(x + w/2, mae_fem, w, color='tomato',    alpha=0.8, label='FEM')
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel('MAE (°C)', fontsize=12)
    ax2.set_title('MAE vs COMSOL', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(list(bars1)+list(bars2), mae_fdm+mae_fem):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{val:.1f}°C', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('validate_comsol_summary.png', dpi=300, bbox_inches='tight')
    print("  Figure sauvegardée : validate_comsol_summary.png")
    plt.show()


# =============================================================================
# Benchmark GP surrogate vs FEM
# =============================================================================
# Note : le GP prédit T_max global (sur 1800s), pas T à t=600s.
# On le compare donc directement contre le FEM sur T_max global —
# c'est cohérent. On ne le met pas dans la validation COMSOL parce que
# COMSOL exporte T à t=600s, ce qui est une quantité différente.

def benchmark_gp(cas_test=None):
    print(f"\n{'='*60}")
    print("BENCHMARK GP SURROGATE vs FEM — T_max global")
    print(f"{'='*60}")
    print("  (comparaison cohérente : les deux prédisent T_max sur 1800s)\n")

    with open('Models/surrogate_gp.pkl', 'rb') as f:
        gp, scaler_X, scaler_y = pickle.load(f)

    if cas_test is None:
        cas_test = [
            (0.3,  0.06, 40000, "k très faible"),
            (0.5,  0.1,  50000, "k faible      "),
            (2.0,  0.08, 60000, "k moyen       "),
            (8.0,  0.1,  70000, "k élevé       "),
            (14.0, 0.05, 35000, "k très élevé  "),
        ]

    from tps_fct_fem import simulation_principale

    print(f"  {'Cas':<16} | {'FEM':>8} | {'GP':>8} | {'±σ':>6} | {'Erreur':>7} | {'Speedup':>9}")
    print("  " + "-"*62)

    T_fem_list, T_gp_list = [], []

    for k, L_val, q_max, nom in cas_test:
        class prm:
            rho=1800; cp=800; k_therm=k; epsilon=0.85

        t0 = time.time()
        Res = simulation_principale(
            [0,L_val],[0,L_val], NX, NY, DT, 1800.0, [],
            T_INIT, T_STRU, T_ENTREE, q_max, SIGMA, prm,
            save_all=True, verbose=False
        )
        t_fem = time.time() - t0
        T_fem = np.max(Res["T_field_complet"])

        t0 = time.time()
        X_new  = scaler_X.transform([[np.log(k), L_val, q_max]])
        y_s, std_s = gp.predict(X_new, return_std=True)
        y_real = scaler_y.inverse_transform(y_s.reshape(-1,1)).ravel()[0]
        T_gp   = np.exp(y_real)
        T_std  = T_gp * std_s[0] * scaler_y.scale_[0]
        t_ml   = time.time() - t0

        err     = abs(T_gp - T_fem) / T_fem * 100
        speedup = t_fem / t_ml
        print(f"  {nom} | {T_fem:>6.0f}°C | {T_gp:>6.0f}°C | "
              f"±{T_std:>3.0f}°C | {err:>6.2f}% | {speedup:>7.0f}×")

        T_fem_list.append(T_fem)
        T_gp_list.append(T_gp)

    T_fem_arr = np.array(T_fem_list)
    T_gp_arr  = np.array(T_gp_list)
    err_all   = np.abs(T_gp_arr - T_fem_arr) / T_fem_arr * 100

    print(f"\n  R²  GP vs FEM : {r2_score(T_fem_arr, T_gp_arr):.4f}")
    print(f"  MAE GP vs FEM : {mean_absolute_error(T_fem_arr, T_gp_arr):.2f} °C")
    print(f"  Erreur moy    : {err_all.mean():.2f}%")
    print(f"  Erreur max    : {err_all.max():.2f}%")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VALIDATION COMSOL vs FDM vs FEM")
    print(f"t = {T_TARGET}s  ·  maillage {NX}×{NY}  ·  dt = {DT}s")
    print("=" * 60)

    # Charger les données COMSOL
    print("\n[1/4] Données COMSOL...")
    k_vals, T_comsol_k = load_comsol_txt('Validation/comsol_variation_k.txt')
    q_vals, T_comsol_q = load_comsol_txt('Validation/comsol_variation_qmax.txt')

    args_k = [(k,   L, Q_MAX, T_TARGET) for k in k_vals]
    args_q = [(0.5, L, q,     T_TARGET) for q in q_vals]

    # FDM
    print(f"\n[2/4] Simulations FDM ({len(k_vals)+len(q_vals)} total)...")
    T_fdm_k = run_parallel(run_fdm_one, args_k)
    T_fdm_q = run_parallel(run_fdm_one, args_q)
    print("  FDM terminé ✅")

    # FEM
    print(f"\n[3/4] Simulations FEM ({len(k_vals)+len(q_vals)} total)...")
    T_fem_k = run_parallel(run_fem_one, args_k)
    T_fem_q = run_parallel(run_fem_one, args_q)
    print("  FEM terminé ✅")

    # Validation FDM/FEM vs COMSOL
    print("\n[4/4] Validation et graphiques...")
    results_k = validate(k_vals, T_comsol_k, T_fdm_k, T_fem_k,
                         param_name='k', param_unit='W/(m·K)',
                         label='variation_k', t_target=T_TARGET)
    results_q = validate(q_vals, T_comsol_q, T_fdm_q, T_fem_q,
                         param_name='q_max', param_unit='W/m²',
                         label='variation_qmax', t_target=T_TARGET)

    plot_summary(results_k, results_q)

    # Résumé terminal
    print("\n" + "=" * 60)
    print("RÉSUMÉ FINAL")
    print("=" * 60)
    for label, res in [('Variation k', results_k), ('Variation q_max', results_q)]:
        print(f"\n  {label} :")
        print(f"    FDM : R²={res['r2_fdm']:.4f}  MAE={res['mae_fdm']:.1f}°C  "
              f"erreur moy {res['err_fdm'].mean():.1f}%")
        print(f"    FEM : R²={res['r2_fem']:.4f}  MAE={res['mae_fem']:.1f}°C  "
              f"erreur moy {res['err_fem'].mean():.1f}%")

    # Benchmark GP séparé
    print("\n")
    benchmark_gp()

    print("\n✅ Validation terminée")