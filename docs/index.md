# 🔬 Retinal Oxygen Transport  
## *Physics-Informed & Numerical Modeling of Diffusion in the Human Retina*

---

## 🧠 Overview

This project provides a comprehensive study of **oxygen transport across the multilayered retina**, integrating:

- ⚙️ **Finite-Difference Method (FDM)**
- 🔲 **Finite-Volume Method (FVM)**
- 🧩 **Inverse Physics-Informed Neural Networks (PINNs)**  
- 💡 (Coming Soon) **Forward PINNs**

Each method solves or learns from the underlying PDE governing oxygen diffusion and consumption within biological tissues. Our focus is on combining **scientific rigor** with **modern machine learning** to advance computational physiology.

---

## 🧪 Scientific Context

The retina relies on precise oxygen delivery to maintain visual function. Disruption in this process is implicated in vision-threatening diseases like **diabetic retinopathy** and **macular degeneration**.

The domain is modeled as **four layers**, each with its own:
- Diffusivity $D_i$
- Reaction rate $k_i$
- Boundary concentrations $C_0$, $C_L$

We solve the PDE:

$$
\\frac{\\partial C}{\\partial t} = D(z) \\frac{\\partial^2 C}{\\partial z^2} - k(z) C
$$

This equation is tackled using both classical numerical methods and data-driven neural solvers.

---

## 🧰 Methods

- **FDM (Forward Euler / Steady-State)**  
  Simple grid-based method using explicit or implicit time integration.

- **FVM (Backward Euler / Steady-State)**  
  Conserves mass over control volumes; stable and robust.

- **Inverse PINN**  
  Learns $D_i$, $k_i$, $C_0$, and $C_L$ directly from profile data using physics-informed loss terms (residuals, boundary continuity, etc).

- **Forward PINN** *(Coming Soon)*  
  Solves the diffusion PDE using DeepXDE or a PyTorch-based collocation solver.

---

## 🖥️ Repository Features

- Modular code for numerical and ML solvers
- Visualization tools for concentration profiles and error metrics
- Transformer-based inverse PINN with multi-headed attention
- Pretraining + physics fine-tuning pipeline
- Support for noisy and synthetic data

---

## 👥 Authors

This project is a collaborative effort by a team of Biomedical Engineering students. See [README](https://github.com/Ziad-Ashraf-Abdu/Retinal_O2_transport#authors--contributions) for full author list.

