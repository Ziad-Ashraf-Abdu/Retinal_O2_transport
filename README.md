# Retinal O₂ Transport: Numerical & Machine‑Learning Methods
## Table of Contents

1. [Project Overview](#project-overview)
2. [Scientific Motivation](#scientific-motivation)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Configuration](#configuration)
7. [Numerical Solution Methods](#numerical-solution-methods)
  * Method of Lines (MOL)

     * Steady‑State Solution (via long-time integration)
     * Time‑Dependent Solver (BDF Solver)
   * Finite‑Difference Method (FDM)

     * Steady‑State Solver
     * Time‑Dependent Solver (Forward Euler)
   * Finite‑Volume Method (FVM)

     * Steady‑State Solver
     * Time‑Dependent Solver (Backward Euler)
8. [Forward PINN Model](#forward-pinn-model)
9. [Inverse PINN Model](#inverse-pinn-model)

   * Data Generation (`MasterpieceDataset`)
   * Network Architecture (`MasterpieceInversePINN`, `EnhancedAttentionBlock`)
   * Training Pipeline (`MasterpieceLightning`)
   * Physics‑Informed Loss Components
10. [COMSOL MODEL](#comsol-validation)
11. [Evaluation & Output](#evaluation--output)
12. [Authors & Contributions](#authors--contributions)
13. [References](#references)

---

## Project Overview
> Our project page https://ziad-ashraf-abdu.github.io/Retinal_O2_transport/

This repository implements and compares four distinct computational schemes for modeling oxygen transport in the retina's layered structure:

1.**Method of Lines (MOL)** – time‑dependent solver (BDF), with steady‑state obtained via long‑time integration.
2. **Finite‑Difference Method (FDM)** – both steady‑state and explicit time‑dependent solvers.
3. **Finite‑Volume Method (FVM)** – both steady‑state and implicit time‑dependent solvers.
4. **Forward Physics‑Informed Neural Network (Forward PINN)** – a mesh‑free, PDE‑embedded neural solver (to be released imminently).
5. **Inverse PINN** – learns layer parameters $(D_i, k_i)$ and boundary concentrations $(C_0, C_L)$ directly from simulated or measured oxygen profiles.

The main goal is to quantify and contrast **accuracy**, **computational cost**, and **data requirements** of classical numerical schemes versus physics‑informed machine learning approaches in a four‑layer retinal setting.

---

## Scientific Motivation

* **Biological Context:** Photoreceptors in the outer retina have high metabolic demand; oxygen must diffuse from vitreous and choroid through four distinct layers, each with unique diffusivity $D_i$ and consumption rate $k_i$.
* **Clinical Relevance:** Impaired transport underlies diabetic retinopathy and age‑related macular degeneration. Fast, accurate parameter estimation could guide personalized therapies.
* **Computational Challenge:** Classical solvers require dense meshes and repeated forward solves for inverse problems. PINNs aim to reduce mesh dependence and enable direct parameter inference.

---

## Repository Structure

```
RETINA/
│
├── Analytical Solution/           # Contains analytical/benchmark results
│   ├── plots/                     # Visualizations of the analytical solutions
│   ├── steady_state_fv.py        # Finite volume solution for steady-state
│   ├── Steady state.py           # Analytical steady-state script
    ├── test.py
|   ├── time dependent_LASTVERSION.py  
│   └── Time dependent.py         # Analytical time-dependent solution
│
├── Inverse Model/                # PINN implementation for the inverse problem
│   ├── lightning_logs/           # Logging from PyTorch Lightning
│   ├── model_checkpoints/        # Saved model checkpoints
│   ├── plots/                    # Model-generated plots
│   └── O2_profile.py             # PINN for inverse oxygen profile modeling
│
├── Forward Model/ 
│   ├── pltos/
    ├── All_layers_Forward_Model.py

├── .gitignore                    # Git ignored files
├── README.md                     # Project overview and instructions
└── requirements.txt              # Python dependencies

```

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Ziad-Ashraf-Abdu/Retinal_O2_transport.git
   cd Retinal_O2_transport
   ```
2. **Create and activate a virtual environment**
   * With venv

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate       # Linux/macOS
   .venv\Scripts\activate.bat      # Windows
   ```
   * With conda
   ```bash
   conda create -n myenv python=3.9
   conda activate myenv
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Numerical Solvers Only

```bash
python Steady state.py #FDM
python Time dependent.py #FDM
python steady state_FVM.py #FVM
python time dependent_LASTVERSION.py #FVM
```

These commands will:

* Discretize the four‑layer domain in $z\in[0,1]$
* Solve the PDE using specified method and mode
* Plot and save results to `figures/num_*`

### 2. Inverse PINN Training

```bash
python O2_profile.py
```

This:

* Generates synthetic profiles via the four‑layer analytical forward solver
* Trains the `Lightning` module through pretraining and physics‑fine‑tuning
* Saves parity plots, reconstructions, and summary metrics
---

## Configuration

All key hyperparameters are defined at the top of `O2_profile.py`:

```python
MAX_RUNS = 5
PRETRAIN_EPOCHS = 200
FINETUNE_EPOCHS = 300
LR_PRETRAIN = 2e-3
LR_FINETUNE = 1e-4
BATCH_SIZE = 64
GRAD_CLIP_VAL = 0.5

z_np = np.linspace(0, 1, 1000, dtype=np.float32)  # Domain discretization
idx_interfaces = [250, 500, 750]                  # Layer boundaries
param_ranges = {
    'D1':(...), 'k1':(...),
    'D2':(...), 'k2':(...),
    'D3':(...), 'k3':(...),
    'D4':(...), 'k4':(...),
    'C0':(...), 'C_L':(...)
}
```

Modify these to suit your computational budget and desired accuracy.

---

## Numerical Solution Methods

### Method of Lines (MOL)

#### Steady‑State Solver

* Discretizes ∂²u/∂z² using central finite differences on a uniform grid.
* Forms an ODE system:  
  du_i/dt = D_i * ( u_{i+1} - 2u_i + u_{i-1} ) / (Δz)²  - k_i * u_i
* Enforces flux continuity at interfaces:  
  u_interface(+) = u_interface(-)
* Applies Dirichlet boundary conditions:  
  u(0, t) = 20 mmHg, u(L, t) = 100 mmHg

#### Time‑Dependent Solver (Backward Differentiation Formula — BDF)

* Time integration using implicit BDF for stiff systems.
* Solves ODE system with solve_ivp(method='BDF'), t ∈ [0, 30 s].
* Stable for large time steps — accurate for reaction-diffusion in layered retina.

### Finite‑Difference Method (FDM)

#### Steady‑State Solver

* Discretizes $\frac{d^2C}{dz^2} - \frac{k(z)}{D(z)}\,C = 0$ with central differences on a uniform grid.
* Enforces flux continuity at interfaces via harmonic means of $D$.
* Solves resulting tridiagonal linear system with Thomas algorithm.

#### Time‑Dependent Solver (Explicit Forward Euler)

* Approximates $\partial_t C = D(z)\,\partial_{zz}C - k(z)\,C$ via Forward Euler and central spatial differences.
* Handles interface diffusion coefficients with harmonic or arithmetic means.
* Simple but subject to CFL stability constraint: $\Delta t \le \Delta z^2/(2\max D)$.

### Finite‑Volume Method (FVM)

#### Steady‑State Solver

* Integrates diffusion–reaction equation over control volumes.
* Applies divergence theorem to get flux differences at cell faces.
* Enforces boundary/tridiagonal continuity via arithmetic means at interfaces.

#### Time‑Dependent Solver (Backward Euler)

* Temporal discretization via implicit backward Euler for unconditional stability.
* Assembles and solves a sparse linear system $A\,C^{n+1}=C^n$ at each timestep.
* Accurate but requires solving large linear systems (sparse LU).

---

## Forward PINN Model

### Physics-Informed Neural Network (PINN) for Modeling Oxygen Diffusion in the Retina

> Here is a working [Kaggle Notebook](https://www.kaggle.com/code/rahmafathyedress/retinalo2-transport) for the 4 Layers code and a [Colab Notebook](https://colab.research.google.com/drive/1bAXA8vFtfz-1vo3hFOHv3sqa66WQnPH3?usp=sharing) 
for the IR model with fixed parameters

#### 1. Objective
The goal of this work is to model the steady-state oxygen diffusion across four anatomical layers of the retina, ensuring both physical and biological accuracy. The layers involved are:
- **Inner Retina (IR)**
- **Outer Retina (OR)**
- **Photoreceptor Layer (FL)**
- **Choroidal Capillary (CC)**

The model integrates both the underlying physics of diffusion and real experimental measurements to capture the physiological behavior of oxygen transport within retinal tissues.

#### 2. Governing Equation
The mathematical foundation of the model is a second-order partial differential equation (PDE) representing diffusion and consumption:

$$
D(z) \cdot \frac{d^2u}{dz^2} - k(z) \cdot u = 0
$$

Where:
- $u(z)$: Oxygen partial pressure at depth $z$
- $D(z)$: Diffusion coefficient (layer-specific)
- $k(z)$: Consumption rate (layer-specific)

The equation form remains the same across all layers, but the parameters $D$ and $k$ change depending on the layer properties.

#### 3. Domain Partitioning Using Logical Masks
To correctly apply layer-specific physics within a single PDE framework, logical masks are used. Each spatial region corresponding to a layer is identified by a boolean mask. For example, the mask for the Inner Retina (IR) is defined as:

```python
mask_IR = tf.cast((z >= zL) & (z <= zIR), tf.float32)
```

The PDE residual is constructed as a sum over the contributions of all layers, weighted by their respective masks:

$$
\text{PDE Residual} = \sum_{i=1}^{4} \text{mask}_i \cdot \left( D_i \cdot u_{zz} - k_i \cdot u \right)
$$

This ensures that each point in the domain follows the correct physical behavior according to its assigned layer.

#### 4. Boundary Conditions and Reference Data
Boundary conditions are defined at the two ends of the domain:
- **Left boundary (z = 0):** Oxygen pressure set to the first measured value in the IR reference data.
- **Right boundary (z = 250):** Oxygen pressure set to the last measured value in the CC reference data.

In addition, 44 internal reference points from experimental measurements are incorporated as soft constraints using `PointSetBC` from DeepXDE, anchoring the neural network solution to biological reality.

#### 5. Neural Network Architecture
The computational model employs a fully connected feedforward neural network (FNN) with the following specifications:
- **Architecture:** [1, 256, 256, 256, 256, 256, 256, 256, 256, 1]
- **Activation Function:** tanh
- **Initialization:** Glorot normal initializer

The network predicts the scalar field $u(z)$, representing the oxygen partial pressure, across the entire retinal depth.

#### 6. Loss Function Definition and Training Process
##### 6.1. Loss Components
The total loss minimized during training is a weighted combination of four terms:

$$
\mathcal{L}_{\text{total}} = w_{\text{PDE}} \cdot \mathcal{L}_{\text{PDE}} + w_{\text{BC1}} \cdot \mathcal{L}_{\text{BC1}} + w_{\text{BC2}} \cdot \mathcal{L}_{\text{BC2}} + w_{\text{Data}} \cdot \mathcal{L}_{\text{Data}}
$$

| Component | Description |
|-----------|-------------|
| $\mathcal{L}_{\text{PDE}}$ | Residual loss of the PDE (physics consistency) |
| $\mathcal{L}_{\text{BC1}}$ | Loss at left boundary (z = 0) |
| $\mathcal{L}_{\text{BC2}}$ | Loss at right boundary (z = 250) |
| $\mathcal{L}_{\text{Data}}$ | MSE between model predictions and reference data |

##### 6.2. How the Loss is Computed
- **PDE Residual Loss ($\mathcal{L}_{\text{PDE}}$):** Calculated as the mean squared difference between the left-hand and right-hand sides of the PDE at collocation points.
- **Boundary Losses ($\mathcal{L}_{\text{BC1}}, \mathcal{L}_{\text{BC2}}$):** Squared error between predicted and true boundary oxygen pressures.
- **Reference Data Loss ($\mathcal{L}_{\text{Data}}$):** Squared error between the model predictions and real empirical measurements.

##### 6.3. Loss Weights Across Training Phases
Training is structured into four adaptive phases with different emphasis on each loss component:

| Phase | Purpose | Loss Weights: [PDE, BC1, BC2, Data] |
|-------|---------|--------------------------------------|
| 1 | Emphasize learning from data | [1.0, 10.0, 10.0, 50.0] |
| 2 | Balance PDE with reference | [1.0, 5.0, 5.0, 20.0] |
| 3 | Fine-tune data accuracy | [1.0, 10.0, 10.0, 30.0] |
| 4 | Final convergence | [1.0, 5.0, 5.0, 25.0] + L-BFGS Optimizer |

##### 6.4. Training Algorithm
- Training utilizes the Adam optimizer for initial convergence, followed by the L-BFGS optimizer for high-precision minimization.
- Automatic differentiation is used to compute the required derivatives for the PDE residual loss.

#### 7. Evaluation Metrics
Model performance is assessed using three primary error metrics:
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Square Error)**
- **Relative L2 Norm (Normalized Euclidean Error)**

#### 8. Results

| Layer                  | MAE (mmHg) | RMSE (mmHg) | Rel L2   |
|------------------------|------------|-------------|----------|
| Inner Retina (IR)      | 0.0724     | 0.0941      | 0.0036   |
| Outer Retina (OR)      | 0.8980     | 1.0554      | 0.0315   |
| Photoreceptor Layer (FL)| 1.1227    | 1.4182      | 0.0258   |
| Choroidal Capillary (CC)| 3.3135    | 3.5869      | 0.0361   |
| **Total Relative Error (average)** |        |             | **2.11%** |

**Interpretation:**
- The Inner Retina (IR) showed the highest accuracy due to its stable profile.
- The Outer Retina (OR) and Photoreceptor Layer (FL) exhibited moderate errors due to more complex dynamics.
- The Choroidal Capillary (CC) showed higher errors but maintained acceptable relative accuracy, especially considering the larger absolute oxygen values.

#### 9. Conclusion
The developed PINN successfully modeled oxygen diffusion in a multi-layered retinal structure using a single PDE structure with spatially dependent physical parameters. Logical masking enabled correct assignment of physics within each anatomical region, and real measurement data improved model fidelity.

By employing a staged, adaptive training strategy with carefully selected loss weightings, the model achieved high accuracy across all layers of interest, supporting its potential for use in advanced retinal physiological simulations.

#### 10. Computational Performance Analysis
In addition to accuracy evaluation, the computational performance of the model was assessed. The average computational time per evaluation point for each anatomical layer was measured, providing insight into the model's efficiency during inference.

| Layer | Avg Time (s) | Time/Point (ms) | Std Dev (s) |
|-------|--------------|-----------------|-------------|
| IR    | 0.000852     | 0.0085          | 0.000057    |
| OR    | 0.000834     | 0.0083          | 0.000037    |
| FL    | 0.000854     | 0.0085          | 0.000047    |
| CC    | 0.000846     | 0.0085          | 0.000047    |
| **Total** | **0.0034** |                 |             |

**Interpretation:**
- The model demonstrated consistent inference times across all layers, with average evaluation time per point ≈ 8.5 ms.
- The small standard deviations confirm that predictions are stable in terms of computational cost regardless of the anatomical region.

---

## Inverse PINN Model

### Data Generation: `MasterpieceDataset`

* **Synthetic Profiles:**

  * Four‑layer analytical solver (`forward_piecewise_analytical`) with robust linear algebra.
  * High‑SNR noise model using signal‑to‑noise ratio sampling.
* **Normalization:**

  * Profile normalization via median & standard deviation.
  * Parameter normalization to $[0,1]$ based on `param_ranges`.

### Network Architecture

#### `EnhancedAttentionBlock`

* Multi‑head self‑attention (batch‑first)
* Feed‑forward residual sublayers with GELU activation
* Learnable residual scaling factors $\alpha_1,\alpha_2$

#### `MasterpieceInversePINN`

1. **Patch Embedding:**

   * Divides 1D input into patches (size= 8); project via `Linear` + positional encoding
2. **Transformer Blocks:**

   * Stacks of `EnhancedAttentionBlock`
3. **Pooling & Heads:**

   * Adaptive avg & max pooling → concatenation → three specialized MLP heads
   * **Diffusion Head:** predicts \\(\[D\_1,D\_2,D\_3,D\_4]\\)
   * **Reaction Head:** predicts \\(\[k\_1,k\_2,k\_3,k\_4]\\)
   * **Boundary Head:** predicts \\(\[C\_0,C\_L]\\)

### Training Pipeline: `MasterpieceLightning`

* **Pretraining Stage:**

  * Optimizes data‐fidelity loss (weighted MSE + Huber + MAE)
  * Early stopping on validation loss
* **Physics Fine‑Tuning:**

  * Adds composite physics loss:

    1. Dirichlet boundary MSE
    2. PDE residuals (finite‑difference second derivatives)
    3. Concentration & flux continuity at three interfaces
    4. Smoothness and parameter‐bound penalties
  * Two‐stage optimizer schedule (AdamW, CosineAnnealingWarmRestarts)
---

## COMSOL Validation


To ensure our numerical and PINN approaches faithfully reproduce the underlying physics—and to guard against subtle mis‑interpretations of the reference paper—we implemented an independent model in **COMSOL Multiphysics®**. This served as a "sanity check" for boundary conditions, layer‑specific parameters, and the transition between time‑dependent and steady‑state behavior.

### 1. Validation Motivation

* **Human fallibility**: Even well‑documented reference studies can be misread or mis‑transcribed.
* **PDE behavior**: Our hypothesis (and literature reports) indicate that the retinal diffusion–reaction PDE behaves as transient within a finite time window, but rapidly approaches steady state thereafter.
* **Cross‑platform confidence**: By reproducing the problem in a commercial finite‑element environment, we gain high assurance in both our parametrization and our own solvers' implementation.

### 2. COMSOL Model Setup

* **Geometry & Layers**: Four contiguous 1D domains matching our Python/DeepXDE definitions (photoreceptor side → vitreous side).
* **Material Properties**: Diffusivities $D_i$ and reaction rates $k_i$ imported directly from our code's parameter file.
* **Boundary & Initial Conditions**:

  * $C(z,0)$ set to baseline profile.
  * Fixed concentrations $C_0$ at the choroidal boundary and $C_L$ at the vitreal surface.
* **Time Study**: Simulated on $t \in [0,\,T_{\text{final}}]$ with fine time sampling to capture initial transients and long‑term asymptote.

### 3. Key Findings

* **Transient Interval**: For $t \lesssim 5$ s (model‑dependent), the solution exhibits clear time‑varying dynamics, matching our FDM/PINN transient runs.
* **Steady‑State Emergence**: Beyond this interval, concentration profiles converge to the same steady‑state solution predicted by our FVM solver (maximum deviation < 0.1 %).
* **Interface Continuity**: COMSOL confirms flux continuity across layer boundaries without spurious jumps—reinforcing our handling of interface conditions in the PINN loss.


### 5. Conclusions

* **Parameter fidelity**: The match between COMSOL and our own codes confirms that our diffusivity/reaction-rate assignments and boundary conditions are correctly implemented.
* **Model regime**: The clear transient‑to‑steady transition validates our choice to solve the time‑dependent PDE only up to the identified interval, then switch to a steady‑state solver for efficiency.
* **Confidence boost**: This cross‑platform agreement underpins all downstream results—whether FDM, FVM, or PINN‑based.

---

## Evaluation & Output

After training, the script automatically:

1. **Generates Figures** (`plots/`):

   * **Parity Plots**: True vs. predicted parameters, ±10 % bands
   * **Profile Reconstructions**: Best, worst, and random sample overlays
   * **Error Analysis**: Run‑wise error evolution, boxplots, correlation heatmaps
2. **Saves Model Checkpoints** (`masterpiece_checkpoints/`)
3. **Writes Summary** (`masterpiece_summary.json`) with best-run metrics

Use these to compare classical vs. PINN performance in terms of:

* **Accuracy** (relative error, RMSE, R²)
* **Compute Time** (CPU vs. GPU, mesh points vs. collocation points)
* **Data Requirements** (synthetic vs. experimental)
* **Ease of Extension** (adding layers, unsteady terms)

---
## Report & Overleaf

You can view and download the full project report and LaTeX code below:

- 📄 *Final Report (Google Drive):* [Project Report Folder](https://drive.google.com/drive/folders/14qoEp9Y48tujJ_jz_XZeNjQWd03Mk0lw)
- 📘 *Overleaf LaTeX Source Code:* [Overleaf Project](https://www.overleaf.com/2454224742hjhsgtvyghmr)

---

## Authors & Contributions

* **Zeyad A. Abdu** – Lead inverse PINN design, assesting in COMSOL implemtation, repo management.
* **Suhila T. Elmasry** – Numerical FVM implementation & validation, Documentation.
* **Rahma F. Hamouda Edress** – Leading forward model design, Documentation.
* **Haneen M. Gameel** – Numerical FVM implementation & validation, Documentation.
* **Ahmed W. A. Naem** – Numerical FDM implementation & validation.
* **Saif M. Ali** – Forward model design, repo management.
* **Ahmed M. Abdelsalam** – Lead COMSOL implementation, assesting in inverse PINN validation.
* **Mohanud H. Abdelnaby** – Documentation, presentation, and repo management.

---

## References


1. W. E. Schiesser, *Differential Equation Analysis in Biomedical Science and Engineering: Case Studies with MATLAB*, Cambridge University Press, 2013, ch. "Retinal O₂ transport."
2. D. Y. Yu and S. J. Cringle, "Oxygen distribution and consumption in the retina: Modeling and measurement," *Bioengineering Department*, University of Illinois at Urbana–Champaign, Tech. Rep., 2002.
3. M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics‐Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations," *Journal of Computational Physics*, vol. 378, pp. 686–707, Feb. 2019.
4. L. Lu, P. Jin, and G. E. Karniadakis, "DeepXDE: A Deep Learning Library for Solving Differential Equations," *SIAM Review*, vol. 63, no. 1, pp. 208–228, 2021.
5. SciANN, "SciANN: A Keras‑based, high‑level API for physics‐informed neural networks," 2024. [Online]. Available: [https://www.sciann.com](https://www.sciann.com)
6. A. Toledo‑Marín *et al.*, "Convolutional Surrogate Modeling of Multiphase Flow in Porous Media," in *Proc. NeurIPS ML for Physical Sciences Workshop*, 2021.
7. J. Seman, "Adaptive Physics‐Informed Neural Networks for Reaction‐Diffusion Systems," *Frontiers in Physics*, vol. 8, p. 632, 2020.
8. H. Y. Zhu *et al.*, "PINN for Piecewise Coefficient PDEs," *ETNA*. vol. 56, pp. 1–27, 2022.
9. J. D. Liu *et al.*, "3D Image‐based Arterial Network Modeling of Retinal Oxygen Tension," *IEEE Trans. Biomedical Engineering*, vol. 56, no. 9, pp. 2345–2354, Sep. 2009.
10. E. Aquah *et al.*, "CFD Modeling of Retinal Ischemia and Hemoglobin Affinity Effects," *Computers in Biology and Medicine*, vol. 131, p. 104234, 2021.
11. E. J. McHugh *et al.*, "3D Finite‐Element Diffusion Modeling of Hypoxia in AMD from OCT Scans," *PLOS One*, vol. 14, no. 6, e0216215, 2019.
12. S. A. Spencer *et al.*, "In Vivo Measurement of Retinal Oxygen Tension Gradients," *Invest. Ophthalmol. Vis. Sci.*, vol. 54, no. 6, pp. 4060–4067, Jun. 2013.
13. A. Xiaowei *et al.*, "Integrating Physics‐Based Modeling with Machine Learning: A Survey," 2020. [Online]. Available: [https://beiyulincs.github.io/teach/fall\_2020/papers/xiaowei.pdf](https://beiyulincs.github.io/teach/fall_2020/papers/xiaowei.pdf)
14. "Mass diffusivity," *Wikipedia*, 2024. [Online]. Available: [https://en.wikipedia.org/wiki/Mass\_diffusivity](https://en.wikipedia.org/wiki/Mass_diffusivity)
15. "Physics‐Based Deep Learning," 2024. [Online]. Available: [https://physicsbaseddeeplearning.org/intro.html](https://physicsbaseddeeplearning.org/intro.html)


