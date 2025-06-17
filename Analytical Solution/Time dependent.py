# Time-Dependent Oxygen Transport in Retina Using Finite Difference

import numpy as np
import matplotlib.pyplot as plt

# =============================
# DOMAIN AND DISCRETIZATION
# =============================
L = 0.8  # Retinal thickness in mm
N = 44   # Number of spatial grid points
x = np.linspace(0, L, N)
dx = x[1] - x[0]

# =============================
# LAYER BOUNDARIES AND LABELS
# =============================
boundaries = [0.0, 0.2, 0.4, 0.6, 0.8]
layer_labels = ["IR", "OR", "FL", "CC"]

# =============================
# PHYSICAL PARAMETERS (D & R)
# =============================
D_vals = {"IR": 1e-2, "OR": 1e-2, "FL": 1e-2, "CC": 1e-2}  # mm^2/s
R_vals = {"IR": 0.1, "OR": 0.1, "FL": 0.1, "CC": 0.1}     # 1/s

# Assign properties per layer
layer = np.empty(N, dtype='<U2')
D = np.zeros(N)
R = np.zeros(N)

for i in range(N):
    xi = x[i]
    if xi <= boundaries[1]:
        layer[i] = "IR"
    elif xi <= boundaries[2]:
        layer[i] = "OR"
    elif xi <= boundaries[3]:
        layer[i] = "FL"
    else:
        layer[i] = "CC"
    D[i] = D_vals[layer[i]]
    R[i] = R_vals[layer[i]]

# =============================
# TIME SETTINGS
# =============================
t_final = 30.0  # seconds
dt = 0.01
n_steps = int(t_final / dt)
time_array = np.linspace(0, t_final, n_steps + 1)

# =============================
# INITIAL CONDITIONS
# =============================
C = np.zeros(N)
C[-1] = 130.0  # Boundary condition at CC (µM)
C_time = np.zeros((n_steps + 1, N))
C_time[0] = C.copy()

# =============================
# TIME-STEPPING LOOP (EXPLICIT)
# =============================
for step in range(1, n_steps + 1):
    C_new = C.copy()

    for i in range(1, N - 1):
        # Handle layer interfaces explicitly (continuity)
        if layer[i] != layer[i - 1] or layer[i] != layer[i + 1]:
            D_left = D[i-1] if layer[i] == layer[i-1] else (D[i] + D[i-1]) / 2
            D_right = D[i+1] if layer[i] == layer[i+1] else (D[i] + D[i+1]) / 2
        else:
            D_left = 2 * D[i] * D[i - 1] / (D[i] + D[i - 1])
            D_right = 2 * D[i] * D[i + 1] / (D[i] + D[i + 1])

        diffusion = (D_right * (C[i + 1] - C[i]) - D_left * (C[i] - C[i - 1])) / dx**2
        C_new[i] = C[i] + dt * (diffusion - R[i] * C[i])

    # Boundary conditions (converted from mmHg to µM using alpha ≈ 1.3)
    C_new[0] = 26.0
    C_new[-1] = 130.0

    C = C_new.copy()
    C_time[step] = C.copy()

    if step % 1000 == 0:
        print(f"Time: {step * dt:.1f} s")

# =============================
# LAYER MASKS
# =============================
IR_mask = (x >= 0.0) & (x < 0.2)
OR_mask = (x >= 0.2) & (x < 0.4)
FL_mask = (x >= 0.4) & (x < 0.6)
CC_mask = (x >= 0.6)
layers = [("IR", IR_mask), ("OR", OR_mask), ("FL", FL_mask), ("CC", CC_mask)]

# =============================
# PLOT: LAYER-WISE EVOLUTION
# =============================
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Oxygen Concentration Evolution in Each Layer", fontsize=16)

for ax, (label, mask) in zip(axs.flat, layers):
    x_layer = x[mask]
    C_layer_time = C_time[:, mask]

    for t_idx in [int(t / dt) for t in [6, 12, 18, 24, 30]]:
        ax.plot(x_layer, C_layer_time[t_idx], label=f"t = {t_idx * dt:.0f}s")

    ax.set_title(f"{label} Layer")
    ax.set_xlabel("Position (mm)")
    ax.set_ylabel("O₂ (µM)")
    ax.grid(True)
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
def relative_error(numerical, reference):
    return (np.linalg.norm(numerical - reference) / np.linalg.norm(reference)) * 100 #Euclidean norm

# === Define reference profiles in µM (from textbook) ===
# True Values from the reference 
ref_IR = np.array([
    26.000, 25.714, 25.519, 25.428, 25.441,
    25.558, 25.779, 26.104, 26.533, 27.066, 27.703
])   # 20–21.3 mmHg × 1.3
ref_OR = np.array([
    27.716, 28.457, 29.315, 30.303, 31.408,
    32.630, 33.982, 35.477, 37.115, 38.909, 40.846
]) # 21.3–31.4
ref_FL = np.array([
    40.846, 42.939, 45.214, 47.671, 50.310,
    53.157, 56.212, 59.501, 63.024, 66.794, 70.824
])# 31.4–54.5
ref_CC = np.array([
    70.837, 75.140, 79.755, 84.695, 89.960,
    95.589, 101.608, 108.030, 114.881, 122.200, 130.000
]) # 54.5–100

# === Extract numerical results ===
C_IR = C[IR_mask]
C_OR = C[OR_mask]
C_FL = C[FL_mask]
C_CC = C[CC_mask]

# === Calculate relative errors ===
err_IR = relative_error(C_IR, ref_IR)
err_OR = relative_error(C_OR, ref_OR)
err_FL = relative_error(C_FL, ref_FL)
err_CC = relative_error(C_CC, ref_CC)

# === Total error across domain ===
ref_full = np.concatenate([ref_IR, ref_OR, ref_FL, ref_CC])
C_full = np.concatenate([C_IR, C_OR, C_FL, C_CC])
total_error = relative_error(C_full, ref_full)

# === Print results ===
print(f"Relative Error IR:  {err_IR:.2f}%")
print(f"Relative Error OR:  {err_OR:.2f}%")
print(f"Relative Error FL:  {err_FL:.2f}%")
print(f"Relative Error CC:  {err_CC:.2f}%")
print(f"Total Relative Error: {total_error:.2f}%")