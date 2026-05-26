"""
ML Surrogate Model pour TPS
Prédit le champ complet T(x,y,t) sans simulation FEM complète

Input  : (k, L, q_max)
Output : T_field (181, 21, 21)
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "Method function"))

from tps_fct_fem import simulation_principale

# ── Constantes ───────────────────────────────────────────────────────
DT           = 10.0
T_FINAL      = 1800.0
N_STEPS      = int(T_FINAL / DT)          # 180
FIELD_SHAPE  = (N_STEPS + 1, 21, 21)      # (181, 21, 21)
CACHE_DIR    = 'sim_cache'
os.makedirs('Models', exist_ok=True)


def run_one_simulation(args):
    i, k, L, q_max = args

    class prm:
        rho     = 1800.0
        cp      = 800.0
        k_therm = k
        epsilon = 0.85

    Res = simulation_principale(
        [0, L], [0, L], 21, 21, DT, T_FINAL,  # dt=10s
        temps_instantanes=[],
        T_initiale=293.15, T_stru=293.15,
        t_entree=1200.0, q_max=q_max,
        sigma=5.67e-8, prm=prm,
        save_all=True, verbose=False
    )

    T_field = Res["T_field_complet"]  # (181, 21, 21) directement — PAS de [::5]
    return T_field                    # ← CORRECTION : suppression du [::5] erroné


def generate_dataset_parallel(n_samples=300,
                               save_file='dataset_TPS.npz'):
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Paramètres — seed fixe pour reproductibilité
    np.random.seed(42)
    k_samples = np.random.uniform(0.2, 15.0, n_samples)
    L_samples = np.random.uniform(0.04, 0.12, n_samples)
    q_samples = np.random.uniform(30000, 80000, n_samples)
    X = np.column_stack([k_samples, L_samples, q_samples])
    np.save(os.path.join(CACHE_DIR, 'X_params.npy'), X)

    t0 = time.time()
    print(f"Génération séquentielle de {n_samples} simulations...")
    print(f"  Shape attendu par sim : {FIELD_SHAPE}")

    for i in range(n_samples):
        cache_file = os.path.join(CACHE_DIR, f'sim_{i:04d}.npy')

        # ── Reprise automatique — skip si déjà calculé ───────────────
        if os.path.exists(cache_file):
            arr = np.load(cache_file)
            if arr.shape == FIELD_SHAPE:
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i+1}/{n_samples} | déjà en cache ✓")
                continue
        # ─────────────────────────────────────────────────────────────

        T_field = run_one_simulation((i, X[i, 0], X[i, 1], X[i, 2]))
        np.save(cache_file, T_field)  # ← sauvegarde individuelle

        if (i + 1) % 10 == 0:
            elapsed   = time.time() - t0
            per_sim   = elapsed / (i + 1)
            remaining = per_sim * (n_samples - i - 1)
            print(f"  Progress: {i+1}/{n_samples} | "
                  f"{elapsed/60:.1f} min écoulées | "
                  f"~{remaining/60:.1f} min restantes")
            _assemble_and_save(n_samples, X, save_file)

    n_valid = _assemble_and_save(n_samples, X, save_file)
    print(f"\n✅ Dataset sauvegardé : {save_file}")
    print(f"   {n_valid} simulations valides")
    data = np.load(save_file)
    print(f"   X: {data['X'].shape} · y: {data['y'].shape}")
    return data['X'], data['y']


def _assemble_and_save(n_samples, X, save_file):
    """Assemble les caches individuels en un seul dataset."""
    X_valid, y_valid = [], []
    for i in range(n_samples):
        cache_file = os.path.join(CACHE_DIR, f'sim_{i:04d}.npy')
        if not os.path.exists(cache_file):
            continue
        arr = np.load(cache_file)
        if arr.shape != FIELD_SHAPE:  # rejet si shape incorrect
            continue
        X_valid.append(X[i])
        y_valid.append(arr)
    if y_valid:
        np.savez_compressed(save_file,
                            X=np.array(X_valid),
                            y=np.array(y_valid))
    return len(y_valid)


def train_surrogate(dataset_file='dataset_TPS.npz',
                    save_model=os.path.join('Models', 'surrogate_model.pkl')):

    print("Chargement dataset...")
    data = np.load(dataset_file)
    X, y = data['X'], data['y']
    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"  k range : [{X[:,0].min():.2f}, {X[:,0].max():.2f}]")
    print(f"  q range : [{X[:,2].min():.0f}, {X[:,2].max():.0f}]")
    print(f"  T range : [{y.min():.1f}, {y.max():.1f}] °C")

    n_samples   = y.shape[0]
    field_shape = y.shape[1:]   # (181, 21, 21)
    y_flat      = y.reshape(n_samples, -1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_flat, test_size=0.2, random_state=42
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s  = scaler_X.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train)
    y_test_s  = scaler_y.transform(y_test)

    model = MLPRegressor(
        hidden_layer_sizes=(1024, 512, 256, 128),  # ← meilleure architecture
        activation='relu',                          # ← gagne sur tanh
        solver='adam',
        learning_rate_init=1e-3,                   # ← meilleur lr
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=True
    )

    print("\nEntraînement...")
    model.fit(X_train_s, y_train_s)

    y_train_pred = scaler_y.inverse_transform(model.predict(X_train_s))
    y_test_pred  = scaler_y.inverse_transform(model.predict(X_test_s))

    r2_train = r2_score(y_train, y_train_pred)
    r2_test  = r2_score(y_test,  y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    print(f"\n  R² train : {r2_train:.4f}")
    print(f"  R² test  : {r2_test:.4f}")
    print(f"  MAE test : {mae_test:.3f} °C")

    with open(save_model, 'wb') as f:
        pickle.dump((model, scaler_X, scaler_y, field_shape), f)
    print(f"\n✅ Modèle sauvegardé : {save_model}")

    return model, scaler_X, scaler_y, field_shape


def predict_champ_complet(k, L, q_max, model_file=os.path.join('Models', 'surrogate_model.pkl')):
    with open(model_file, 'rb') as f:
        data = pickle.load(f)

    if len(data) == 4:
        model, scaler_X, scaler_y, field_shape = data
    else:
        model, scaler_X, scaler_y = data
        field_shape = FIELD_SHAPE

    X_new    = np.array([[k, L, q_max]])
    X_scaled = scaler_X.transform(X_new)
    y_scaled = model.predict(X_scaled)
    y_pred   = scaler_y.inverse_transform(y_scaled)
    T_field  = y_pred.reshape(field_shape)

    n_steps = field_shape[0]
    temps   = np.linspace(0, T_FINAL, n_steps)  # cohérent avec field_shape

    print(f"k={k}, L={L}, q_max={q_max}")
    print(f"T_max prédit: {np.max(T_field):.1f}°C")

    return T_field, temps


def benchmark_speed():

    class prm:
        rho     = 1800.0
        cp      = 800.0
        k_therm = 0.5
        epsilon = 0.85

    k, L, q_max = 0.8, 0.1, 50000

    start = time.time()
    Res = simulation_principale(
        [0, L], [0, L], 21, 21, DT, T_FINAL,
        temps_instantanes=[],
        T_initiale=293.15, T_stru=293.15,
        t_entree=1200.0, q_max=q_max,
        sigma=5.67e-8, prm=prm,
        save_all=True, verbose=False
    )
    time_fem = time.time() - start
    T_max_fem = np.max(Res["T_field_complet"])

    start = time.time()
    T_field_pred, _ = predict_champ_complet(k, L, q_max)
    time_ml = time.time() - start
    T_max_ml = np.max(T_field_pred)

    speedup = time_fem / time_ml
    error   = abs(T_max_ml - T_max_fem)

    print(f"\n  FEM (dt={DT}s):")
    print(f"    T_max : {T_max_fem:.2f} °C")
    print(f"    Temps : {time_fem:.3f} s")
    print(f"\n  Surrogate ML:")
    print(f"    T_max : {T_max_ml:.2f} °C")
    print(f"    Temps : {time_ml:.6f} s")
    print(f"\n  Speedup : {speedup:.0f}×")
    print(f"  Erreur  : {error:.2f}°C ({error/T_max_fem*100:.2f}%)")


RUN_FEM_BENCHMARK = False   # mettre True pour activer le benchmark FEM (~2 min)

if __name__ == "__main__":
    print("=" * 70)
    print("ML SURROGATE MODEL POUR TPS")
    print("=" * 70)

    # print("\n[1/3] Generation dataset...")
    # generate_dataset_parallel(n_samples=300, save_file='dataset_TPS.npz')

    # print("\n[2/3] Entrainement modele...")
    # train_surrogate()

    if RUN_FEM_BENCHMARK:
        print("\n[3/3] Benchmark vitesse...")
        benchmark_speed()
    else:
        print("\n[3/3] Test prédiction rapide (sans FEM)...")
        T_field, temps = predict_champ_complet(k=0.8, L=0.1, q_max=50000)
        print(f"  T_max prédit : {T_field.max():.1f}°C  |  shape : {T_field.shape}")

    print("\n" + "=" * 70)
    print("ML SURROGATE MODEL TERMINE")
    print("=" * 70)
    print("\nUtilisation:")
    print("  from ml_surrogate1 import predict_champ_complet")
    print("  T_field, temps = predict_champ_complet(k=0.5, L=0.1, q_max=50000)")