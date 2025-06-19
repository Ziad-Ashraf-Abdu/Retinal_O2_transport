import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt

dde.config.set_default_float("float32")
torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------
# Domain parameters
zL, zIR = 0.0, 100.0  
D_min, D_max = 1000.0, 50050.0
k_min, k_max = 0.01, 0.505
PIR_S = 20.0  

# -------------------------
# Normalization functions
def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

def denormalize(x_norm, min_val, max_val):
    return x_norm * (max_val - min_val) + min_val

# -------------------------
# Geometry definition
class CustomGeometry(dde.geometry.Geometry):
    def __init__(self):
        bbox = np.array([[zL, 0.0, 0.0], [zIR, 1.0, 1.0]], dtype=np.float32)
        diam = np.linalg.norm(bbox[1] - bbox[0])
        super().__init__(3, bbox, diam)

    def inside(self, x):
        return np.ones((x.shape[0],), dtype=bool)

    def on_boundary(self, x):
        return np.logical_or(np.isclose(x[:, 0], zL), np.isclose(x[:, 0], zIR))

    def random_points(self, n, random="pseudo"):
        z = np.random.uniform(zL, zIR, (n, 1)).astype(np.float32)
        D = np.random.uniform(D_min, D_max, (n, 1)).astype(np.float32)
        k = np.random.uniform(k_min, k_max, (n, 1)).astype(np.float32)
        D_norm = normalize(D, D_min, D_max)
        k_norm = normalize(k, k_min, k_max)
        return np.hstack((z, D_norm, k_norm)).astype(np.float32)

    def random_boundary_points(self, n, random="pseudo"):
        n_half = n // 2
        z = np.concatenate([np.full(n_half, zL), np.full(n - n_half, zIR)]).reshape(-1, 1).astype(np.float32)
        D = np.random.uniform(D_min, D_max, (n, 1)).astype(np.float32)
        k = np.random.uniform(k_min, k_max, (n, 1)).astype(np.float32)
        D_norm = normalize(D, D_min, D_max)
        k_norm = normalize(k, k_min, k_max)
        return np.hstack((z, D_norm, k_norm)).astype(np.float32)

geom = CustomGeometry()

# -------------------------
# PDE definition
def pde(x, u):
    z, D_norm, k_norm = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    D = denormalize(D_norm, D_min, D_max)
    k = denormalize(k_norm, k_min, k_max)
    d2u_dz2 = dde.grad.hessian(u, x, i=0, j=0)
    return D * d2u_dz2 - k * u

# -------------------------
# Boundary conditions
def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], zL)

def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], zIR)

bc_left = dde.icbc.DirichletBC(geom, lambda x: PIR_S * np.ones_like(x[:, 0:1]), boundary_left)
bc_right = dde.icbc.DirichletBC(geom, lambda x: 15.0 * np.ones_like(x[:, 0:1]), boundary_right)

# -------------------------
# Data preparation
data = dde.data.PDE(geom, pde, [bc_left, bc_right], num_domain=15000, num_boundary=3000)

# -------------------------
# Neural network definition
net = dde.nn.FNN([3] + [256] * 6 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# -------------------------
# Training
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=40000)

model.compile("L-BFGS")
losshistory, train_state = model.train()

# -------------------------
# Testing the solution
z_test = np.linspace(zL, zIR, 200).reshape(-1, 1)
D_test = np.full_like(z_test, D_max)
k_test = np.full_like(z_test, k_max)
D_norm_test = normalize(D_test, D_min, D_max)
k_norm_test = normalize(k_test, k_min, k_max)
X_test = np.hstack((z_test, D_norm_test, k_norm_test))
u_pred = model.predict(X_test)

# -------------------------
# Analytical solution
def exact_solution(z, D, k):
    alpha = np.sqrt(k / D)
    A = (PIR_S - 15.0) / (1 - np.exp(-alpha * zIR))
    return PIR_S - A * (1 - np.exp(-alpha * z))

u_exact = exact_solution(z_test.flatten(), D_max, k_max).reshape(-1, 1)

# -------------------------
# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(z_test, u_pred, lw=2, label="PINN Prediction")
plt.plot(z_test, u_exact, "--", lw=2, label="Analytical Solution")
plt.xlabel("z (μm)")
plt.ylabel("O2 Concentration (mm Hg)")
plt.title("Steady-State O2 in Inner Retina")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# Display statistics
mse = np.mean((u_pred - u_exact) ** 2)
max_error = np.max(np.abs(u_pred - u_exact))
rel_error = np.mean(np.abs((u_exact - u_pred) / u_exact)) * 100
print(f"Best Train Loss: {train_state.best_loss_train:.4e}")    
print(f"Best Test Loss: {train_state.best_loss_test:.4e}")
print(f"MSE: {mse:.4e}")
print(f"Max Error: {max_error:.4e}")
print(f"Rel Error: {rel_error:.4e}")
