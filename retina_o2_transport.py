# Retina O₂ Transport – Fixed Version to Match Target Graphs

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Grid setup
nir = nor = nfl = ncc = 11
zl_ir = zl_or = zl_fl = zl_cc = 200
dz = zl_ir / (nir - 1)

# Diffusion parameters (same for all layers)
Dir = Dor = Dfl = Dcc = 1.0e4  # diffusivity (µm²/s)

# Different consumption rates for each layer (key to matching the graphs!)
kir = 0.05   # Inner retina - low consumption (nearly flat profile)
kor = 0.02   # Outer retina - very low consumption (slight increase)
kfl = 0.08   # Fluid layer - moderate consumption (steady increase)
kcc = 0.15   # Choroid - higher consumption (steep increase)

# Boundary conditions
pir_start = 20.0   # O₂ at inner retina start (vitreous side)
pcc_end = 100.0    # O₂ at choroid end (blood supply)

# Spatial grids
zg_ir = np.linspace(0, zl_ir, nir)
zg_or = np.linspace(0, zl_or, nor)
zg_fl = np.linspace(0, zl_fl, nfl)
zg_cc = np.linspace(0, zl_cc, ncc)

# Better initial conditions based on expected steady-state profiles
def initial_conditions():
    u0 = np.zeros(nir + nor + nfl + ncc)

    # IR: nearly flat around 20 mmHg (low metabolism) - NO NOISE
    u0[0:nir] = 20.0

    # OR: slight increase from ~15 to ~25 mmHg - NO NOISE
    z_norm = np.linspace(0, 1, nor)
    u0[nir:nir+nor] = 15 + 10 * z_norm

    # FL: steady increase from ~25 to ~50 mmHg - NO NOISE
    z_norm = np.linspace(0, 1, nfl)
    u0[nir+nor:nir+nor+nfl] = 25 + 25 * z_norm

    # CC: steep increase from ~50 to 100 mmHg - NO NOISE
    z_norm = np.linspace(0, 1, ncc)
    u0[nir+nor+nfl:] = 50 + 50 * z_norm

    return u0

u0 = initial_conditions()

# 2nd derivative with proper boundary handling
def second_derivative_with_coupling(u_layer, u_prev, u_next, dz, is_first=False, is_last=False):
    d2u = np.zeros_like(u_layer)
    n = len(u_layer)

    if is_first:
        d2u[0] = 0
        d2u[1:-1] = (u_layer[2:] - 2*u_layer[1:-1] + u_layer[:-2]) / dz**2
        if u_next is not None:
            d2u[-1] = (u_next[0] - 2*u_layer[-1] + u_layer[-2]) / dz**2
        else:
            d2u[-1] = 0
    elif is_last:
        if u_prev is not None:
            d2u[0] = (u_layer[1] - 2*u_layer[0] + u_prev[-1]) / dz**2
        else:
            d2u[0] = 0
        d2u[1:-1] = (u_layer[2:] - 2*u_layer[1:-1] + u_layer[:-2]) / dz**2
        d2u[-1] = 0
    else:
        if u_prev is not None:
            d2u[0] = (u_layer[1] - 2*u_layer[0] + u_prev[-1]) / dz**2
        else:
            d2u[0] = 0
        d2u[1:-1] = (u_layer[2:] - 2*u_layer[1:-1] + u_layer[:-2]) / dz**2
        if u_next is not None:
            d2u[-1] = (u_next[0] - 2*u_layer[-1] + u_layer[-2]) / dz**2
        else:
            d2u[-1] = 0

    return d2u

# System of ODEs with proper layer coupling
def pde_system(t, u):
    uir = u[0:nir].copy()
    uor = u[nir:nir+nor].copy()
    ufl = u[nir+nor:nir+nor+nfl].copy()
    ucc = u[nir+nor+nfl:].copy()

    uir[0] = pir_start
    ucc[-1] = pcc_end

    uir_t = Dir * second_derivative_with_coupling(uir, None, uor, dz, is_first=True) - kir * uir
    uor_t = Dor * second_derivative_with_coupling(uor, uir, ufl, dz) - kor * uor
    ufl_t = Dfl * second_derivative_with_coupling(ufl, uor, ucc, dz) - kfl * ufl
    ucc_t = Dcc * second_derivative_with_coupling(ucc, ufl, None, dz, is_last=True) - kcc * ucc

    uir_t[0] = 0
    ucc_t[-1] = 0

    return np.concatenate([uir_t, uor_t, ufl_t, ucc_t])

t_span = (0, 30)
t_eval = np.array([0, 5, 10, 20, 30])
sol = solve_ivp(pde_system, t_span, u0, method='BDF', t_eval=t_eval, rtol=1e-6, atol=1e-8)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
solutions_ir = sol.y[0:nir, :]
solutions_or = sol.y[nir:nir+nor, :]
solutions_fl = sol.y[nir+nor:nir+nor+nfl, :]
solutions_cc = sol.y[nir+nor+nfl:, :]

for i in range(len(sol.t)):
    color = plt.cm.viridis(i / (len(sol.t) - 1))
    axs[0, 0].plot(zg_ir, solutions_ir[:, i], color=color, linewidth=1.5)
    axs[0, 1].plot(zg_or, solutions_or[:, i], color=color, linewidth=1.5)
    axs[1, 0].plot(zg_fl, solutions_fl[:, i], color=color, linewidth=1.5)
    axs[1, 1].plot(zg_cc, solutions_cc[:, i], color=color, linewidth=1.5)

titles = ['Inner, t = 0,5,...,30 s,\nD = 1 × 10⁴ μm²/s',
          'Outer, t = 0,10,...,30 s,\nD = 1 × 10⁴ μm²/s',
          'Fluid, t = 0,10,...,30 s,\nD = 1 × 10⁴ μm²/s',
          'Choroid, t = 0,10,...,30 s,\nD = 1 × 10⁴ μm²/s']

xlabels = ['zg_ir, μm', 'zg_or, μm', 'zg_fl, μm', 'zg_cc, μm']
ylabels = ['uir(z,t), mmHg O₂', 'uor(z,t), mmHg O₂', 'ufl(z,t), mmHg O₂', 'uco(z,t), mmHg O₂']

for i, ax in enumerate(axs.flat):
    ax.set_title(titles[i], fontsize=10)
    ax.set_xlabel(xlabels[i])
    ax.set_ylabel(ylabels[i])
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Final steady-state values:")
print(f"IR: {solutions_ir[:, -1].min():.1f} - {solutions_ir[:, -1].max():.1f} mmHg")
print(f"OR: {solutions_or[:, -1].min():.1f} - {solutions_or[:, -1].max():.1f} mmHg")
print(f"FL: {solutions_fl[:, -1].min():.1f} - {solutions_fl[:, -1].max():.1f} mmHg")
print(f"CC: {solutions_cc[:, -1].min():.1f} - {solutions_cc[:, -1].max():.1f} mmHg")
