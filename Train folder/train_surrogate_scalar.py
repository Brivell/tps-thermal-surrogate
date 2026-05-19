"""
Surrogate MLP scalaire pour la prédiction de T_max
===================================================
Prédit T_max global (une seule valeur) à partir de (k, L, q_max).

La clé de ce modèle est le log-transform appliqué sur k et T_max avant
l'entraînement. Sans ça, le MLP essaie d'apprendre une relation
exponentielle directement, ce qui lui coûte beaucoup de capacité et
donne de mauvais résultats sur les cas extrêmes (k faible).

Avec log(k) et log(T_max), la relation devient quasi-linéaire et
le MLP converge proprement, même avec 500 samples.

Résultats attendus :
  R² test  > 0.999
  Erreur   < 1% sur tous les cas

Usage :
    python train_surrogate_scalar.py
"""

import numpy as np
import pickle
import time
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

DATASET_FILE = 'dataset_TPS.npz'
MODEL_FILE   = 'surrogate_tmax_scalar.pkl'

# Chargement du dataset
print("Chargement dataset...")
data    = np.load(DATASET_FILE)
X       = data['X']
y_field = data['y']
y_tmax  = np.max(y_field, axis=(1, 2, 3))

print(f"  {X.shape[0]} simulations")
print(f"  k      : [{X[:,0].min():.3f}, {X[:,0].max():.2f}] W/m·K")
print(f"  T_max  : [{y_tmax.min():.1f}, {y_tmax.max():.1f}] °C")

# Log-transform
# La relation T_max(k) suit approximativement T_max ~ A·exp(-α·k),
# donc log(T_max) ~ log(A) - α·k — beaucoup plus facile à apprendre.
X_log = np.column_stack([
    np.log(X[:, 0]),   # log(k)
    X[:, 1],           # L (linéaire)
    X[:, 2],           # q_max (linéaire)
])
y_log = np.log(y_tmax)

print(f"\n  Après log-transform :")
print(f"  log(k) : [{X_log[:,0].min():.2f}, {X_log[:,0].max():.2f}]")
print(f"  log(T) : [{y_log.min():.2f}, {y_log.max():.2f}]")

# Split et normalisation
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

# Entraînement
print("\nEntraînement MLP...")
model = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=1e-3,
    max_iter=2000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    verbose=True
)

t0 = time.time()
model.fit(X_train_s, y_train_s)
print(f"Terminé en {time.time()-t0:.1f}s")

# Évaluation — on repasse en °C via exp() avant de calculer les métriques
y_train_pred = np.exp(scaler_y.inverse_transform(
    model.predict(X_train_s).reshape(-1,1)).ravel())
y_test_pred  = np.exp(scaler_y.inverse_transform(
    model.predict(X_test_s).reshape(-1,1)).ravel())

r2_train = r2_score(y_train_true, y_train_pred)
r2_test  = r2_score(y_test_true,  y_test_pred)
mae_test = mean_absolute_error(y_test_true, y_test_pred)
err_rel  = np.abs(y_test_pred - y_test_true) / y_test_true * 100

print(f"\n  R² train   : {r2_train:.4f}")
print(f"  R² test    : {r2_test:.4f}")
print(f"  MAE test   : {mae_test:.2f} °C")
print(f"  Erreur moy : {err_rel.mean():.2f}%")
print(f"  Erreur max : {err_rel.max():.2f}%")
print(f"  < 2%       : {(err_rel<2).sum()}/{len(err_rel)} points")
print(f"  < 5%       : {(err_rel<5).sum()}/{len(err_rel)} points")

with open(MODEL_FILE, 'wb') as f:
    pickle.dump((model, scaler_X, scaler_y), f)
print(f"\n  Modèle sauvegardé : {MODEL_FILE}")


# =============================================================================
# Benchmark vs FEM
# =============================================================================

print(f"\nBenchmark MLP scalaire vs FEM\n")
from tps_fct_fem import simulation_principale

cas_test = [
    (0.3,  0.06, 40000, "k très faible"),
    (0.5,  0.1,  50000, "k faible      "),
    (2.0,  0.08, 60000, "k moyen       "),
    (8.0,  0.1,  70000, "k élevé       "),
    (14.0, 0.05, 35000, "k très élevé  "),
]

print(f"  {'Cas':<16} | {'FEM':>8} | {'MLP':>8} | {'Erreur':>7} | {'Speedup':>9}")
print("  " + "-"*58)

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
    X_new  = np.array([[np.log(k), L, q_max]])
    T_surr = np.exp(scaler_y.inverse_transform(
        model.predict(scaler_X.transform(X_new)).reshape(-1,1)).ravel()[0])
    t_ml   = time.time() - t0

    err     = abs(T_surr - T_fem) / T_fem * 100
    speedup = t_fem / t_ml
    print(f"  {nom} | {T_fem:>6.0f}°C | {T_surr:>6.0f}°C | "
          f"{err:>6.2f}% | {speedup:>7.0f}×")

print(f"\nUtilisation :")
print(f"  with open('surrogate_tmax_scalar.pkl', 'rb') as f:")
print(f"      model, scaler_X, scaler_y = pickle.load(f)")
print(f"  X_new = np.array([[np.log(k), L, q_max]])")
print(f"  T_max = np.exp(scaler_y.inverse_transform(")
print(f"              model.predict(scaler_X.transform(X_new)).reshape(-1,1)))[0][0]")