import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Define layer properties
layers = [
    {"name": "IR", "L": 200, "D": 1e4, "k": 0, "N": 21},
    {"name": "OR", "L": 200, "D": 1e4, "k":0, "N": 21},
    {"name": "FL", "L": 200, "D": 1e4, "k": 0, "N": 21},
    {"name": "CC", "L": 200, "D": 1e4, "k": 0, "N": 21},
]

# Construct global grid
z_all, D_all, k_all = [], [], []
layer_indices = []

for layer in layers:
    N = layer["N"]
    dz = layer["L"] / (N - 1)
    z = np.linspace(0, layer["L"], N)
    layer_indices.append((len(z_all), len(z_all) + len(z)))
    z_all.extend(z)
    D_all.extend([layer["D"]] * len(z))
    k_all.extend([layer["k"]] * len(z))

z = np.array(z_all)
D = np.array(D_all)
k = np.array(k_all)
dz = z[1] - z[0]
N_total = len(z)

# FVM coefficients
main_diag = np.zeros(N_total)
upper_diag = np.zeros(N_total - 1)
lower_diag = np.zeros(N_total - 1)
rhs = np.zeros(N_total)

# Internal nodes
for i in range(1, N_total - 1):
    De = (D[i] + D[i + 1]) / 2
    Dw = (D[i] + D[i - 1]) / 2
    main_diag[i] = (De + Dw) / dz**2 + k[i]
    upper_diag[i] = -De / dz**2
    lower_diag[i - 1] = -Dw / dz**2

# Boundary conditions
main_diag[0] = 1
rhs[0] = 20  # Left Dirichlet BC
main_diag[-1] = 1
rhs[-1] = 100  # Right Dirichlet BC

# Sparse matrix and solve
A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')
u = spsolve(A, rhs)

# Plot each layer
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for i, (layer, (start, end)) in enumerate(zip(layers, layer_indices)):
    z_layer = z[start:end]
    u_layer = u[start:end]
    axs[i].plot(z_layer, u_layer, '-o', label=f"{layer['name']} O₂")
    axs[i].set_title(f"{layer['name']} Layer")
    axs[i].set_xlabel("z (μm)")
    axs[i].set_ylabel("O₂ Pressure (mm Hg)")
    axs[i].grid(True)
    axs[i].set_ylim(0, 110)
    axs[i].legend()

plt.suptitle("Steady-State O₂ Profiles in Each Retinal Layer (FVM)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Define layer properties
layers = [
    {"name": "IR", "L": 200, "D": 1e4, "k": 0.1, "N": 11},
    {"name": "OR", "L": 200, "D": 1e4, "k": 0.1, "N": 11},
    {"name": "FL", "L": 200, "D": 1e4, "k": 0.1, "N": 11},
    {"name": "CC", "L": 200, "D": 1e4, "k": 0.1, "N": 11},
]

# Construct global grid
z_all, D_all, k_all = [], [], []
layer_indices = []

for layer in layers:
    N = layer["N"]
    dz = layer["L"] / (N - 1)
    z = np.linspace(0, layer["L"], N)
    if z_all:
        z = z[1:]  # avoid duplicating interface point
    layer_indices.append((len(z_all), len(z_all) + len(z)))
    z_all.extend(z)
    D_all.extend([layer["D"]] * len(z))
    k_all.extend([layer["k"]] * len(z))

z = np.array(z_all)
D = np.array(D_all)
k = np.array(k_all)
dz = z[1] - z[0]
N_total = len(z)

# FVM coefficients
main_diag = np.zeros(N_total)
upper_diag = np.zeros(N_total - 1)
lower_diag = np.zeros(N_total - 1)
rhs = np.zeros(N_total)

# Internal nodes
for i in range(1, N_total - 1):
    De = (D[i] + D[i + 1]) / 2
    Dw = (D[i] + D[i - 1]) / 2
    main_diag[i] = (De + Dw) / dz**2 + k[i]
    upper_diag[i] = -De / dz**2
    lower_diag[i - 1] = -Dw / dz**2

# Boundary conditions
main_diag[0] = 1
rhs[0] = 20  # Left Dirichlet BC
main_diag[-1] = 1
rhs[-1] = 100  # Right Dirichlet BC

# Sparse matrix and solve
A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')
u = spsolve(A, rhs)

# Plot each layer
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for i, (layer, (start, end)) in enumerate(zip(layers, layer_indices)):
    z_layer = z[start:end]
    u_layer = u[start:end]
    axs[i].plot(z_layer, u_layer, '-o', label=f"{layer['name']} O₂")
    axs[i].set_title(f"{layer['name']} Layer")
    axs[i].set_xlabel("z (μm)")
    axs[i].set_ylabel("O₂ Pressure (mm Hg)")
    axs[i].grid(True)
    axs[i].set_ylim(0, 110)
    axs[i].legend()

plt.suptitle("Steady-State O₂ Profiles in Each Retinal Layer (FVM)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Define layer properties
layers = [
    {"name": "IR", "L": 200, "D": 1e4, "k": 0.1, "N": 11},
    {"name": "OR", "L": 200, "D": 1e4, "k": 0.1, "N": 11},
    {"name": "FL", "L": 200, "D": 1e4, "k": 0.1, "N": 11},
    {"name": "CC", "L": 200, "D": 1e4, "k": 0.1, "N": 11},
]

# Construct global grid
z_all, D_all, k_all = [], [], []
layer_indices = []

for layer in layers:
    N = layer["N"]
    dz = layer["L"] / (N - 1)
    z = np.linspace(0, layer["L"], N)
    layer_indices.append((len(z_all), len(z_all) + len(z)))
    z_all.extend(z)
    D_all.extend([layer["D"]] * len(z))
    k_all.extend([layer["k"]] * len(z))

z = np.array(z_all)
D = np.array(D_all)
k = np.array(k_all)
dz = z[1] - z[0]
N_total = len(z)

# FVM coefficients
main_diag = np.zeros(N_total)
upper_diag = np.zeros(N_total - 1)
lower_diag = np.zeros(N_total - 1)
rhs = np.zeros(N_total)

# Internal nodes
for i in range(1, N_total - 1):
    De = (D[i] + D[i + 1]) / 2
    Dw = (D[i] + D[i - 1]) / 2
    main_diag[i] = (De + Dw) / dz**2 + k[i]
    upper_diag[i] = -De / dz**2
    lower_diag[i - 1] = -Dw / dz**2

# Boundary conditions
main_diag[0] = 1
rhs[0] = 20  # Left Dirichlet BC
main_diag[-1] = 1
rhs[-1] = 100  # Right Dirichlet BC

# Sparse matrix and solve
A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')
u = spsolve(A, rhs)
u_final=u.copy()
# ERROR ANALYSIS
 
 # === Convert final result to micromolar ===
u_micromolar = u_final * 1.3

# === Extract layer-wise data (numerical) ===
start_IR, end_IR = layer_indices[0]
start_OR, end_OR = layer_indices[1]
start_FL, end_FL = layer_indices[2]
start_CC, end_CC = layer_indices[3]

C_IR = u_micromolar[start_IR:end_IR]
C_OR = u_micromolar[start_OR:end_OR]
C_FL = u_micromolar[start_FL:end_FL]
C_CC = u_micromolar[start_CC:end_CC]

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
err_IR = relative_error(C_IR, ref_IR)
err_OR = relative_error(C_OR, ref_OR)
err_FL = relative_error(C_FL, ref_FL)
err_CC = relative_error(C_CC, ref_CC)

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

 # Plot each layer
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for i, (layer, (start, end)) in enumerate(zip(layers, layer_indices)):
    z_layer = z[start:end]
    u_layer = u[start:end]
    axs[i].plot(z_layer, u_layer, '-o', label=f"{layer['name']} O₂")
    axs[i].set_title(f"{layer['name']} Layer")
    axs[i].set_xlabel("z (μm)")
    axs[i].set_ylabel("O₂ Pressure (mm Hg)")
    axs[i].grid(True)
    axs[i].set_ylim(0, 110)
    axs[i].legend()

plt.suptitle("Steady-State O₂ Profiles in Each Retinal Layer (FVM)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
