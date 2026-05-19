import numpy as np

# Fixed physical constants
RHO = 1800.0       # kg/m³
CP = 800.0         # J/(kg·K)
EPSILON = 0.85
SIGMA_SB = 5.67e-8 # W/(m²·K⁴)
T_STRUCT = 293.15  # K (structural temperature)

# Parameter ranges
K_MIN, K_MAX = 0.2, 15.0        # W/(m·K)
L_MIN, L_MAX = 0.04, 0.12       # m
Q_MIN, Q_MAX = 30_000, 80_000   # W/m²
T_SIM = 1800                     # s
NX = NY = 21
NT = 181

K_DEFAULT = 2.0
L_DEFAULT = 0.1
Q_DEFAULT = 50_000
T_THRESHOLD_DEFAULT = 1500

FIELD_SHAPE = (NT, NX, NY)
TIMES = np.linspace(0, T_SIM, NT)
