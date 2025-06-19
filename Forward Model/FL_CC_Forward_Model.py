## FL Layer INtegrated with CC

import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt

# ================================
# CONFIGURATION
# ================================
dde.config.set_default_float("float32")
torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================================
# CONSTANTS & PARAMETERS
# ================================
zL, zIR, zOR = 0.0, 100.0, 150.0  # Spatial domain (z)
D_min, D_max = 1000.0, 50050.0    # Diffusivity constants
P_a = 90.0                        # Boundary O2 pressure

# ================================
# GEOMETRIES
# ================================
geom_fl = dde.geometry.Interval(zL, zIR)
geom_cc = dde.geometry.Interval(zIR, zOR)

# ================================
# PDE DEFINITIONS
# ================================
def diffusivity(z):
    return D_min + (D_max - D_min) * (z - zL) / (zIR - zL)

def pde_fl(x, y):
    z = x[:, 0:1]
    D = diffusivity(z)
    dy_dz = dde.grad.jacobian(y, x, i=0, j=0)
    dy_dz2 = dde.grad.hessian(y, x, i=0, j=0)
    return dy_dz * dde.grad.jacobian(D, x, i=0, j=0) + D * dy_dz2

def pde_cc(x, y):
    dy_dz2 = dde.grad.hessian(y, x, i=0, j=0)
    return D_max * dy_dz2

# ================================
# BOUNDARY CONDITIONS
# ================================
def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], zL)

def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], zOR)

bc_left = dde.DirichletBC(geom_fl, lambda x: P_a, boundary_left)
bc_right = dde.NeumannBC(geom_cc, lambda x: 0, boundary_right)

# ================================
# NEURAL NETWORKS
# ================================
net_fl = dde.maps.FNN([1] + [50]*3 + [1], "tanh", "Glorot normal")
net_cc = dde.maps.FNN([1] + [50]*3 + [1], "tanh", "Glorot normal")

# ================================
# DATASETS
# ================================
ndata = 150
data_fl = dde.data.PDE(geom_fl, pde_fl, [bc_left], num_domain=ndata, num_boundary=10, solution=None)
data_cc = dde.data.PDE(geom_cc, pde_cc, [bc_right], num_domain=ndata, num_boundary=10, solution=None)

model_fl = dde.Model(data_fl, net_fl)
model_fl.compile("adam", lr=1e-3)

model_cc = dde.Model(data_cc, net_cc)
model_cc.compile("adam", lr=1e-3)

# ================================
# STEP 1: Train fluid layer first
# ================================
print("Training Fluid Layer...")
losshistory_fl, train_state_fl = model_fl.train(iterations=10000)

# Predict value and derivative at z = zIR
zIR_array = np.array([[zIR]])
u_interface = model_fl.predict(zIR_array)

dy_dz_interface = model_fl.predict(zIR_array, operator=lambda x, y: dde.grad.jacobian(y, x))

# ================================
# STEP 2: Apply interface conditions on CC layer
# ================================

# Continuity condition: u_cc(zIR) = u_interface
interface_continuity = dde.icbc.PointSetBC(zIR_array, u_interface)

# Flux continuity condition: D_FL(zIR) * du_FL/dz = D_CC * du_CC/dz
D_interface = diffusivity(zIR_array)
target_flux = D_interface * dy_dz_interface

flux_continuity = dde.icbc.OperatorBC(
    geom_cc,
    lambda x, y, _: D_max * dde.grad.jacobian(y, x) - target_flux,
    lambda x, on_boundary: on_boundary and np.isclose(x[0], zIR)
)

# Redefine data for CC layer with the interface conditions
data_cc = dde.data.PDE(
    geom_cc, pde_cc, [bc_right, interface_continuity, flux_continuity],
    num_domain=ndata, num_boundary=10, solution=None
)

model_cc = dde.Model(data_cc, net_cc)
model_cc.compile("adam", lr=1e-3)

print("Training Cell Cluster Layer...")
losshistory_cc, train_state_cc = model_cc.train(iterations=10000)

# ================================
# PREDICTION & VISUALIZATION
# ================================
z_test_fl = np.linspace(zL, zIR, 300)[:, None]
z_test_cc = np.linspace(zIR, zOR, 300)[:, None]

u_pred_fl = model_fl.predict(z_test_fl)
u_pred_cc = model_cc.predict(z_test_cc)

plt.figure(figsize=(10, 6))
plt.plot(z_test_fl, u_pred_fl, label="Fluid Layer (FL)")
plt.plot(z_test_cc, u_pred_cc, label="Cell Cluster (CC)")
plt.axvline(zIR, color='k', linestyle='--', label="Interface (zIR)")
plt.xlabel("z")
plt.ylabel("Oxygen Pressure")
plt.legend()
plt.title("Oxygen Pressure Profile with Flux Continuity")
plt.grid(True)
plt.show()