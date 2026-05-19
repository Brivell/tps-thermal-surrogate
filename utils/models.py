import pickle
import numpy as np
import streamlit as st
from utils.physics import FIELD_SHAPE, TIMES


@st.cache_resource
def load_gp():
    for path in ("Models/surrogate_gp.pkl", "surrogate_gp.pkl"):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            continue
    raise FileNotFoundError("surrogate_gp.pkl not found in Models/ or root")


@st.cache_resource
def load_mlp():
    for path in ("Models/surrogate_model.pkl", "surrogate_model.pkl"):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            continue
    raise FileNotFoundError("surrogate_model.pkl not found in Models/ or root")


@st.cache_resource
def load_dataset():
    for path in ("dataset_TPS.npz",):
        try:
            return np.load(path, allow_pickle=True)
        except FileNotFoundError:
            continue
    raise FileNotFoundError("dataset_TPS.npz not found")


def predict_tmax_gp(k, L, q_max):
    """Returns T_max (°C) and uncertainty (°C)."""
    gp, scaler_X, scaler_y = load_gp()
    X_new = scaler_X.transform([[np.log(k), L, q_max]])
    y_log, std_log = gp.predict(X_new, return_std=True)
    y_real = scaler_y.inverse_transform(y_log.reshape(-1, 1)).ravel()[0]
    T_max = np.exp(y_real)
    T_std = T_max * std_log[0] * scaler_y.scale_[0]
    return float(T_max), float(T_std)


def predict_field(k, L, q_max):
    """Returns T_field (181, 21, 21) and time array (181,)."""
    model_mlp, scaler_X_mlp, scaler_y_mlp, field_shape = load_mlp()
    X_new = np.array([[k, L, q_max]])
    X_scaled = scaler_X_mlp.transform(X_new)
    y_scaled = model_mlp.predict(X_scaled)
    y_pred = scaler_y_mlp.inverse_transform(y_scaled)
    T_field = y_pred.reshape(field_shape)
    return T_field, TIMES


@st.cache_data
def compute_design_space(q_max, n=40):
    """Grid of (k, L) predictions for contour map."""
    k_vals = np.logspace(np.log10(0.2), np.log10(15.0), n)
    L_vals = np.linspace(0.04, 0.12, n)
    K, L_grid = np.meshgrid(k_vals, L_vals)
    T_grid = np.zeros_like(K)
    for i in range(n):
        for j in range(n):
            T_grid[i, j], _ = predict_tmax_gp(K[i, j], L_grid[i, j], q_max)
    return K, L_grid, T_grid
