"""
Surrogate Gaussian Process pour la prédiction de T_max
=======================================================
Entraîne un GP sur le dataset FEM et le compare à un MLP scalaire.

Le GP est préféré au MLP pour ce problème parce qu'avec 500 samples,
il interpole mieux entre les points d'entraînement et donne en plus
un intervalle de confiance (±σ) sur chaque prédiction — ce qui est
très utile pour les décisions de design en ingénierie.

Le log-transform sur k et T_max est essentiel : la relation T_max(k)
est exponentielle, et le log la rend linéaire, ce qui simplifie
beaucoup le problème d'apprentissage.

Usage :
    python train_surrogate_gp.py
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "Method function"))

from tps_fct_fem import simulation_principale

DATASET_FILE       = 'dataset_TPS.npz'
MODEL_GP           = os.path.join('Models', 'surrogate_gp.pkl')
RUN_FEM_BENCHMARK  = False
os.makedirs('Models', exist_ok=True)
os.makedirs('images', exist_ok=True)   # mettre True pour activer le benchmark FEM (~5 min)

# Chargement et préparation du dataset
print("Chargement dataset...")
data    = np.load(DATASET_FILE)
X       = data['X']
y_field = data['y']
y_tmax  = np.max(y_field, axis=(1, 2, 3))

print(f"  {X.shape[0]} simulations")
print(f"  k      : [{X[:,0].min():.3f}, {X[:,0].max():.2f}] W/m·K")
print(f"  T_max  : [{y_tmax.min():.1f}, {y_tmax.max():.1f}] °C")

# Log-transform : T_max(k) est exponentiel, log(T_max)(log(k)) est linéaire
X_log = np.column_stack([np.log(X[:,0]), X[:,1], X[:,2]])
y_log = np.log(y_tmax)

X_train, X_test, y_train, y_test = train_test_split(
    X_log, y_log, test_size=0.2, random_state=42
)
_, _, y_train_true, y_test_true = train_test_split(
    X_log, y_tmax, test_size=0.2, random_state=42
)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_test_s  = scaler_X.transform(X_test)
y_train_s = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()


# =============================================================================
# Gaussian Process
# =============================================================================
# Kernel Matérn ν=2.5 : standard pour les fonctions physiques (deux fois
# différentiables). Le ConstantKernel gère l'amplitude globale.

print("\nEntraînement GP...")
kernel = ConstantKernel(1.0) * Matern(
    length_scale=[1.0, 1.0, 1.0],
    length_scale_bounds=(1e-3, 1e3),
    nu=2.5
)
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,
    n_restarts_optimizer=5,
    normalize_y=True,
    random_state=42
)

t0 = time.time()
gp.fit(X_train_s, y_train_s)
t_gp = time.time() - t0
print(f"Terminé en {t_gp:.1f}s")
print(f"Kernel : {gp.kernel_}")

# Prédiction + incertitude
y_pred_log_s, y_std_s = gp.predict(X_test_s, return_std=True)
y_pred_log = scaler_y.inverse_transform(y_pred_log_s.reshape(-1,1)).ravel()
y_pred_gp  = np.exp(y_pred_log)

# Propagation de l'incertitude : si log(T) = μ ± σ, alors T ≈ exp(μ) · σ
y_std_log = y_std_s * scaler_y.scale_[0]
y_std_gp  = y_pred_gp * y_std_log

r2_gp  = r2_score(y_test_true, y_pred_gp)
mae_gp = mean_absolute_error(y_test_true, y_pred_gp)
err_gp = np.abs(y_pred_gp - y_test_true) / y_test_true * 100

print(f"\n  R² test         : {r2_gp:.4f}")
print(f"  MAE test        : {mae_gp:.2f} °C")
print(f"  Erreur moy      : {err_gp.mean():.2f}%")
print(f"  Erreur max      : {err_gp.max():.2f}%")
print(f"  Points < 2%     : {(err_gp<2).sum()}/{len(err_gp)}")
print(f"  Incertitude moy : ±{y_std_gp.mean():.1f}°C")

with open(MODEL_GP, 'wb') as f:
    pickle.dump((gp, scaler_X, scaler_y), f)
print(f"\n  Modèle sauvegardé : {MODEL_GP}")


# =============================================================================
# MLP scalaire (pour comparaison)
# =============================================================================

print("\nEntraînement MLP scalaire (comparaison)...")
mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=1e-3,
    max_iter=2000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    verbose=False
)
scaler_y_mlp  = StandardScaler()
y_train_s_mlp = scaler_y_mlp.fit_transform(y_train.reshape(-1,1)).ravel()

t0 = time.time()
mlp.fit(X_train_s, y_train_s_mlp)
t_mlp = time.time() - t0

y_pred_mlp = np.exp(scaler_y_mlp.inverse_transform(
    mlp.predict(X_test_s).reshape(-1,1)).ravel())

r2_mlp  = r2_score(y_test_true, y_pred_mlp)
mae_mlp = mean_absolute_error(y_test_true, y_pred_mlp)
err_mlp = np.abs(y_pred_mlp - y_test_true) / y_test_true * 100

print(f"  R² test    : {r2_mlp:.4f}")
print(f"  MAE test   : {mae_mlp:.2f} °C")
print(f"  Erreur moy : {err_mlp.mean():.2f}%")
print(f"  Erreur max : {err_mlp.max():.2f}%")


# =============================================================================
# Résumé comparatif
# =============================================================================

print(f"\n{'Métrique':<20} {'GP':>10} {'MLP':>10}")
print("-" * 42)
print(f"{'R² test':<20} {r2_gp:>10.4f} {r2_mlp:>10.4f}")
print(f"{'MAE (°C)':<20} {mae_gp:>10.2f} {mae_mlp:>10.2f}")
print(f"{'Erreur moy (%)':<20} {err_gp.mean():>10.2f} {err_mlp.mean():>10.2f}")
print(f"{'Erreur max (%)':<20} {err_gp.max():>10.2f} {err_mlp.max():>10.2f}")
print(f"{'Points < 2%':<20} {(err_gp<2).sum():>9}/{len(err_gp)} "
      f"{(err_mlp<2).sum():>9}/{len(err_mlp)}")
print(f"{'Incertitude':<20} {'Oui':>10} {'Non':>10}")
print(f"{'Temps entraîn.':<20} {t_gp:>9.1f}s {t_mlp:>9.1f}s")


# =============================================================================
# Graphiques (avant le benchmark FEM pour un accès rapide)
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Surrogate GP vs MLP — Comparaison sur le dataset de test',
             fontsize=14, fontweight='bold')

ax = axes[0]
lims = [y_test_true.min()-50, y_test_true.max()+50]
ax.plot(lims, lims, 'k--', lw=1.5, label='Parfait (y=x)')
ax.scatter(y_test_true, y_pred_mlp, s=30, color='steelblue', alpha=0.6,
           label=f'MLP · MAE={mae_mlp:.0f}°C', zorder=4)
ax.errorbar(y_test_true, y_pred_gp, yerr=2*y_std_gp,
            fmt='o', color='tomato', alpha=0.6, ms=4,
            elinewidth=0.8, capsize=2,
            label=f'GP · MAE={mae_gp:.0f}°C (±2σ)', zorder=5)
ax.set_xlabel('T_max FEM (°C)', fontsize=12)
ax.set_ylabel('T_max prédit (°C)', fontsize=12)
ax.set_title('Parité FEM vs surrogates', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(y_test_true, err_gp,  s=30, color='tomato',    alpha=0.7,
           label=f'GP  (moy {err_gp.mean():.2f}%)')
ax.scatter(y_test_true, err_mlp, s=30, color='steelblue', alpha=0.7,
           label=f'MLP (moy {err_mlp.mean():.2f}%)')
ax.axhline(5, color='k', linestyle='--', lw=1.5, label='Seuil 5%')
ax.axhline(2, color='g', linestyle='--', lw=1.5, label='Seuil 2%')
ax.set_xlabel('T_max FEM (°C)', fontsize=12)
ax.set_ylabel('Erreur relative (%)', fontsize=12)
ax.set_title('Erreur relative en fonction de T_max', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join('images', 'benchmark_gp_vs_mlp.png'), dpi=300, bbox_inches='tight')
print(f"\n  Figure sauvegardée : images/benchmark_gp_vs_mlp.png")
plt.close()

# =============================================================================
# Benchmark vs FEM sur cas représentatifs
# =============================================================================

if RUN_FEM_BENCHMARK:
    print(f"\nBenchmark GP vs FEM — T_max global\n")

    cas_test = [
        (0.3,  0.06, 40000, "k très faible"),
        (0.5,  0.1,  50000, "k faible      "),
        (2.0,  0.08, 60000, "k moyen       "),
        (8.0,  0.1,  70000, "k élevé       "),
        (14.0, 0.05, 35000, "k très élevé  "),
    ]

    print(f"  {'Cas':<16} | {'FEM':>8} | {'GP':>8} | {'±σ':>6} | {'Erreur':>7} | {'Speedup':>9}")
    print("  " + "-"*65)

    for k, L, q_max, nom in cas_test:
        class prm:
            rho=1800; cp=800; k_therm=k; epsilon=0.85

        t0 = time.time()
        Res = simulation_principale(
            [0,L],[0,L], 21, 21, 10.0, 1800.0, [],
            293.15, 293.15, 1200.0, q_max, 5.67e-8, prm,
            save_all=True, verbose=False
        )
        t_fem = time.time() - t0
        T_fem = np.max(Res["T_field_complet"])

        t0 = time.time()
        X_new    = scaler_X.transform([[np.log(k), L, q_max]])
        y_s, std_s = gp.predict(X_new, return_std=True)
        y_real   = scaler_y.inverse_transform(y_s.reshape(-1,1)).ravel()[0]
        T_gp     = np.exp(y_real)
        T_std    = T_gp * std_s[0] * scaler_y.scale_[0]
        t_ml     = time.time() - t0

        err     = abs(T_gp - T_fem) / T_fem * 100
        speedup = t_fem / t_ml
        print(f"  {nom} | {T_fem:>6.0f}°C | {T_gp:>6.0f}°C | "
              f"±{T_std:>3.0f}°C | {err:>6.2f}% | {speedup:>7.0f}×")
else:
    print("\n  Benchmark FEM ignoré (RUN_FEM_BENCHMARK=False).")

print("\nUtilisation du GP :")
print("  with open('surrogate_gp.pkl', 'rb') as f:")
print("      gp, scaler_X, scaler_y = pickle.load(f)")
print("  X_new = scaler_X.transform([[np.log(k), L, q_max]])")
print("  T_log, T_std = gp.predict(X_new, return_std=True)")
print("  T_max = np.exp(scaler_y.inverse_transform(T_log.reshape(-1,1)))[0][0]")