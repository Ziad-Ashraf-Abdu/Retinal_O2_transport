# Importing required libraries for numerical computation and plotting
import numpy as np
import matplotlib.pyplot as plt

# ================================
# SETUP: Domain and Discretization
# ================================

L = 0.8        # Total retinal thickness in mm (from reference model)
N = 44         # Number of spatial points (matches 11 per layer × 4 layers)
x = np.linspace(0, L, N)  # Create evenly spaced spatial grid over 0.8 mm
dx = x[1] - x[0]          # Spatial step size

# ====================================
# DEFINE: Layer boundaries and labels
# ====================================

boundaries = [0.0, 0.2, 0.4, 0.6, 0.8]          # Start/end of each layer
layer_labels = ["IR", "OR", "FL", "CC"]        # Layer names

# ========================================
# DEFINE: Diffusion and consumption rates
# ========================================

# Diffusion coefficients (mm²/s) for each layer — from reference (constant here)
D_vals = {"IR": 1e-2, "OR": 1e-2, "FL": 1e-2, "CC": 1e-2}

# Oxygen consumption rates (1/s) — from reference model (ncase = 2)
R_vals = {"IR": 0.1, "OR": 0.1, "FL": 0.1, "CC": 0.1}

# ====================================================
# INITIALIZE: Assign layer types and physical values
# ====================================================

layer = np.empty(N, dtype='<U2')  # Create array to hold layer labels
D = np.zeros(N)                   # Create array for diffusion values
R = np.zeros(N)                   # Create array for consumption values

# Loop through all points and assign properties based on position
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
    D[i] = D_vals[layer[i]]  # Assign D based on layer
    R[i] = R_vals[layer[i]]  # Assign R based on layer

# =====================================
# INITIALIZE: Oxygen concentration (µM)
# =====================================

C = np.zeros(N)*0  # Initial guess for concentration (all zeros)

# ========================================================
# FIND: Indices at layer interfaces for special treatment
# ========================================================

interface_indices = [np.argmin(np.abs(x - b)) for b in boundaries[1:-1]]
# Get indices closest to internal boundaries (used for harmonic averaging)

# ================================================
# SOLVE: Iterative finite difference for steady state
# ================================================

tol = 1e-6       # Convergence criterion (L2 norm)
max_iter = 10000 # Maximum number of iterations allowed

# Begin iteration loop
for _ in range(max_iter):
    C_new = C.copy()  # Copy the previous solution

    for i in range(1, N - 1):  # Skip boundaries
        if i in interface_indices:
            # Handle discontinuity in D at interfaces using arithmetic mean
            D_left = D[i - 1]
            D_right = D[i]
            Q_i = R[i] * C[i]
            C_new[i] = (2 * D_right * C[i + 1] + 2 * D_left * C[i - 1] - 2 * dx**2 * Q_i) / (2 * (D_left + D_right))
        else:
            # Standard central difference with harmonic mean diffusivity
            D_left = 2 * D[i] * D[i - 1] / (D[i] + D[i - 1])
            D_right = 2 * D[i] * D[i + 1] / (D[i] + D[i + 1])
            Q_i = R[i] * C[i]
            C_new[i] = (D_left * C[i - 1] + D_right * C[i + 1] - dx**2 * Q_i) / (D_left + D_right)

    # Apply Dirichlet boundary conditions (from reference data)
    C_new[0] = 26     # µM (20 mmHg × 1.3) — PIR-S
    C_new[-1] = 130   # µM (100 mmHg × 1.3) — PCC-S

    # Convergence check (L2 norm of update difference)
    if np.linalg.norm(C_new - C, ord=2) < tol:
        print(f"Converged in {_} iterations.")
        break

    C = C_new.copy()  # Update for next iteration

# =====================
# PLOT: Full profile
# =====================

plt.plot(x, C, label="Oxygen Concentration (µM)", color="blue")
plt.xlabel("Retinal Depth (mm)")
plt.ylabel("Oxygen Concentration (µM)")
plt.title("Steady-State Oxygen Transport in Retina")
plt.grid(True)

# Draw layer boundaries
for b in boundaries:
    plt.axvline(x=b, color='gray', linestyle='--', alpha=0.3)

# Label each region in the plot
for i in range(4):
    mid = (boundaries[i] + boundaries[i + 1]) / 2
    plt.text(mid, np.max(C) * 0.95, layer_labels[i], ha='center', fontsize=10)

plt.tight_layout()
plt.show()

# =================================
# DEFINE: Boolean masks for layers
# =================================

IR_mask = (x >= 0.0) & (x < 0.2)   # IR layer mask
OR_mask = (x >= 0.2) & (x < 0.4)   # OR layer mask
FL_mask = (x >= 0.4) & (x < 0.6)   # FL layer mask
CC_mask = (x >= 0.6)               # CC layer mask

# ===================================
# PLOT: Subplots for each layer
# ===================================

fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # 2x2 layout for each layer
fig.suptitle('Oxygen Concentration by Layer', y=1.02)

axs[0,0].plot(x[IR_mask], C[IR_mask], 'b-', linewidth=2)
axs[0,0].set_title("Inner Retina (IR)")
axs[0,0].set_ylabel("O₂ (µM)")
axs[0,0].grid(True)

axs[0,1].plot(x[OR_mask], C[OR_mask], 'g-', linewidth=2)
axs[0,1].set_title("Outer Retina (OR)")
axs[0,1].grid(True)

axs[1,0].plot(x[FL_mask], C[FL_mask], 'm-', linewidth=2)
axs[1,0].set_title("Fused Layer (FL)")
axs[1,0].set_xlabel("Position (mm)")
axs[1,0].set_ylabel("O₂ (µM)")
axs[1,0].grid(True)

axs[1,1].plot(x[CC_mask], C[CC_mask], 'r-', linewidth=2)
axs[1,1].set_title("Choriocapillaris (CC)")
axs[1,1].set_xlabel("Position (mm)")
axs[1,1].grid(True)

plt.tight_layout()
plt.show()

# ===================================
# COMPARE WITH REFERENCE SOLUTION
# ===================================
def relative_error(numerical, reference):
    return np.linalg.norm(numerical - reference) / np.linalg.norm(reference) * 100 #Euclidean norm

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

