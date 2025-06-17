import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.cm as cm

# PARAMETERS
ncase = 2  # 1: no metabolism, 2: with metabolism

layers = [
    {"name": "IR", "L": 200, "D": 1e4, "k": 0.1, "N": 12},
    {"name": "OR", "L": 200, "D": 1e4, "k": 0.1, "N": 12},
    {"name": "FL", "L": 200, "D": 1e4, "k": 0.1, "N": 12},
    {"name": "CC", "L": 200, "D": 1e4, "k": 0.1, "N": 12},  # ✅ كان فيها N = 1 بالغلط
]

total_time = 30
dt = 0.1
output_times = np.arange(0, total_time + 1, 5)
left_BC = {"value": 20}
right_BC = {"value": 100}

# MESH CONSTRUCTION
z_all, D_all, k_all, layer_indices = [], [], [], []
current_z = 0

for idx, layer in enumerate(layers):
    N = layer["N"]
    dz_layer = layer["L"] / (N - 1)
    
    if idx == 0:
        z_local = np.linspace(0, layer["L"], N)
    else:
        z_local = np.linspace(0,layer["L"],N)
    z_local = z_local + current_z
    layer_indices.append((len(z_all), len(z_all) + len(z_local)))
    z_all.extend(z_local)
    D_all.extend([layer["D"]] * len(z_local))
    k_all.extend([layer["k"]] * len(z_local))
    current_z = z_local[-1]

z = np.array(z_all)
D = np.array(D_all)
k = np.array(k_all)
dz = z[1] - z[0]
N_total = len(z)

interface_points = {start for i, (start, _) in enumerate(layer_indices) if i > 0}

# BUILD MATRIX
def build_matrix(D, k, dz, N_total, dt):
    main_diag = np.zeros(N_total)
    upper_diag = np.zeros(N_total - 1)
    lower_diag = np.zeros(N_total - 1)
    for i in range(1, N_total - 1):
        De = 2 * D[i] * D[i + 1] / (D[i] + D[i + 1])
        Dw = 2 * D[i] * D[i - 1] / (D[i] + D[i - 1])
        main_diag[i] = (De + Dw) / dz**2 + k[i] + 1 / dt
        upper_diag[i] = -De / dz**2
        lower_diag[i - 1] = -Dw / dz**2
        if i in interface_points:
            main_diag[i] += 1e-3
            upper_diag[i] -= 1e-3
    main_diag[0] = 1
    main_diag[-1] = 1
    return diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format="csr")

# INITIAL CONDITION
def custom_initial_condition(z):
    u0 = np.zeros_like(z)

    # IR: 20 → 10
    start, end = layer_indices[0]
    z_local = z[start:end] - z[start]
    u0[start:end] = 20 - (z_local / 200) * 10

    # OR: 10 → 25
    start, end = layer_indices[1]
    z_local = z[start:end] - z[start]
    u0[start:end] = 13 + (z_local / 200) * 15*0.7

    # FL: 25 → 45
    start, end = layer_indices[2]
    z_local = z[start:end] - z[start]
    u0[start:end] = 20 + (z_local / 200) * 20 *1.2

    # CC: 45 → 70
    start, end = layer_indices[3]
    z_local = z[start:end] - z[start]
    u0[start:end] = 46 + (z_local / 200) * 50
    return u0
 # TIME LOOP
A = build_matrix(D, k, dz, N_total, dt)
u = custom_initial_condition(z)
solutions = [(0.0, u.copy())]

for t in np.arange(dt, total_time + dt, dt):
    rhs = u / dt
    rhs[0] = left_BC["value"]
    rhs[-1] = right_BC["value"]
    u = spsolve(A, rhs)
    if np.any(np.isclose(t, output_times)):
        solutions.append((t, u.copy()))
u_final=u.copy()
 # === Convert final result to micromolar ===
u_micromolar = u_final * 1.3

# === Extract layer-wise data (numerical) ===
start_IR, end_IR = layer_indices[0]
start_OR, end_OR = layer_indices[1]
start_FL, end_FL = layer_indices[2]
start_CC, end_CC = layer_indices[3]

C_IR = u_micromolar[start_IR:start_IR+11]
C_OR = u_micromolar[start_OR:start_OR+11]
C_FL = u_micromolar[start_FL:start_FL+11]
C_CC = u_micromolar[start_CC:start_CC+11]
# === Reference values in micromolar ===
ref_IR = np.array([
    26.000, 25.714, 25.519, 25.428, 25.441, 25.558,
    25.779, 26.104, 26.533, 27.066, 27.703
])
ref_OR = np.array([
    27.716, 28.457, 29.315, 30.303, 31.408, 32.630,
    33.982, 35.477, 37.115, 38.909, 40.846
])
ref_FL = np.array([
    40.846, 42.939, 45.214, 47.671, 50.310, 53.157,
    56.212, 59.501, 63.024, 66.794, 70.824
])
ref_CC = np.array([
    70.837, 75.140, 79.755, 84.695, 89.960, 95.589,
    101.608, 108.030, 114.881, 122.200, 130.000
])

# === Relative Error Function ===
def relative_error(numerical, reference):
    return (np.linalg.norm(numerical - reference) / np.linalg.norm(reference)) * 100  # %

# === Compute errors ===
err_IR = relative_error(C_IR,ref_IR)
err_OR = relative_error(C_OR,ref_OR)
err_FL = relative_error(C_FL,ref_FL)
err_CC = relative_error(C_CC,ref_CC)

# === Total domain error ===
ref_full = np.concatenate([ref_IR, ref_OR, ref_FL, ref_CC])
C_full = np.concatenate([C_IR, C_OR, C_FL, C_CC])
total_error = relative_error(C_full, ref_full)

# === Print results ===
print("=========== Relative Errors Compared to True Values (micromolar) ===========")
print(f"IR Error : {err_IR:.2f}%")
print(f"OR Error : {err_OR:.2f}%")
print(f"FL Error : {err_FL:.2f}%")
print(f"CC Error : {err_CC:.2f}%")
print(f"Total Relative Error : {total_error:.2f}%")
print("============================================================================")
 
# PLOTTING
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()
colors = cm.viridis(np.linspace(0.1, 0.9, len(solutions)))

for i, (layer, (start, end)) in enumerate(zip(layers, layer_indices)):
    z_local = z[start:end] - z[start]
    for (t, sol), color in zip(solutions, colors):
        style = '--' if t == 0 else '-'
        axs[i].plot(z_local, sol[start:end], color=color, linestyle=style, label=f't={int(t)}s')
    axs[i].set_title(f"{layer['name']} Layer")
    axs[i].set_xlabel("Position (μm)")
    axs[i].set_ylabel("O₂ Pressure (mm Hg)")
    axs[i].grid(True)
    axs[i].set_xlim(0, layer["L"])
    axs[i].set_ylim(0, 120)
    axs[i].legend()

plt.suptitle(f"Transient O₂ Diffusion (D={layers[0]['D']} μm²/s)", fontsize=14)
plt.tight_layout()
plt.show()