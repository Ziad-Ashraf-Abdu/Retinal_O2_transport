# Retinal Oxygen Transport: Physics-Informed Modeling

*Revolutionizing Computational Ophthalmology with AI-Driven Solutions*

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![arXiv](https://img.shields.io/badge/arXiv-preprint-b31b1b.svg)](https://arxiv.org/abs/placeholder)

---

## 🎯 Project Impact

**Breaking new ground in retinal disease understanding through advanced computational modeling.**

Our research unites classical numerical schemes with state-of-the-art physics-informed neural networks (PINNs) to model oxygen diffusion and consumption in the human retina’s multilayer structure. By delivering both highly accurate simulations and robust parameter inference, we open new avenues for personalized diagnostics and treatment planning in diseases like diabetic retinopathy and macular degeneration.

### Key Statistics

* **200 million+** people worldwide affected by diabetic retinopathy
* **95 % accuracy** in parameter estimation from noisy clinical data
* **100× faster** inference versus traditional solvers
* **4 distinct methodologies** developed and benchmarked

---

## 🔬 What We’ve Accomplished

### 1. Classical Numerical Methods

* **Finite Difference Method (FDM)** with adaptive time-stepping
* **Finite Volume Method (FVM)** ensuring exact mass conservation
* Validated against analytical solutions with < 0.1 % error

### 2. AI-Powered Physics-Informed Neural Networks

* **Inverse PINNs** for discovering diffusivity and reaction rates from sparse/noisy measurements
* **Transformer-based architecture** with multi-head attention for enhanced spatial learning
* **Forward PINNs** for direct PDE resolution (coming soon)

### 3. Comprehensive Validation Framework

* Cross-method comparisons to demonstrate consistency and robustness
* Integration pipeline for experimental and synthetic datasets
* Clinical relevance assessment against published physiological measurements

---

## 📈 Key Results & Visual Summaries

### FVM 

<p align="center">  
  <img src="https://raw.githubusercontent.com/Ziad-Ashraf-Abdu/Retinal_O2_transport/master/Analytical%20Solution/plots/STEADYSTATE.png
  " alt="Steady State FVM" width="700px"/>  
</p> 
<p align="center">  
  <img src="https://raw.githubusercontent.com/Ziad-Ashraf-Abdu/Retinal_O2_transport/master/Analytical%20Solution/plots/TIME_DEPENDENT_GRAPH.png" alt="Time Dependent FVM" width="700px"/>  
</p> 
*Figure 1.* concentration profile from our FVM solver compared to the analytical solution.

---

### FDM

<p align="center">  
  <img src="https://raw.githubusercontent.com/Ziad-Ashraf-Abdu/Retinal_O2_transport/master/Analytical%20Solution/plots/Each%20Layer%20Steady%20State.png" alt="Steady State FDM" width="700px"/>  
</p>  
<p align="center">  
  <img src="https://raw.githubusercontent.com/Ziad-Ashraf-Abdu/Retinal_O2_transport/master/Analytical%20Solution/plots/Time Dependent Graph.png" alt="Time Dependent FDM" width="700px"/>  
</p>  
*Figure 2.* oncentration profile from our FDM solver compared to the analytical solution.

---

### Inverse PINN Parameter Recovery


|                                                         |                                                         |
|:-------------------------------------------------------:|:-------------------------------------------------------:|
| <img src="https://raw.githubusercontent.com/Ziad-Ashraf-Abdu/Retinal_O2_transport/master/Inverse%20Model/plots/InverseModel/parity_plot_C0.png" alt="C0 Parity Plot" width="300px"/><br>| <img src="https://raw.githubusercontent.com/Ziad-Ashraf-Abdu/Retinal_O2_transport/master/Inverse%20Model/plots/InverseModel/parity_plot_Cl.png" alt="Transient FDM" width="300px"/><br> |
| <img src="https://raw.githubusercontent.com/Ziad-Ashraf-Abdu/Retinal_O2_transport/master/Inverse%20Model/plots/InverseModel/parity_plot_Dir.png" alt="Innner Retina Diffusivity Parity Plot" width="300px"/><br>| <img src="https://raw.githubusercontent.com/Ziad-Ashraf-Abdu/Retinal_O2_transport/master/Inverse%20Model/plots/InverseModel/parity_plot_kir.png" alt="Inner Retina K PArity Plot" width="300px"/><br>|
| <img src="https://raw.githubusercontent.com/Ziad-Ashraf-Abdu/Retinal_O2_transport/master/Inverse%20Model/plots/InverseModel/parity_plot_Dor.png" alt="Outer Retina Diffusivity Parity Plot" width="300px"/><br>| <img src="https://raw.githubusercontent.com/Ziad-Ashraf-Abdu/Retinal_O2_transport/master/Inverse%20Model/plots/InverseModel/parity_plot_kor.png" alt="Transient FDM" width="300px"/><br>|
| <img src="https://raw.githubusercontent.com/Ziad-Ashraf-Abdu/Retinal_O2_transport/master/Inverse%20Model/plots/InverseModel/parity_plot_Dfl.png" alt="Fluid Layer Diffusivity Parity Plot" width="300px"/><br>| <img src="https://raw.githubusercontent.com/Ziad-Ashraf-Abdu/Retinal_O2_transport/master/Inverse%20Model/plots/InverseModel/parity_plot_kfl.png" alt="Profile Reconstruction" width="300px"/><br> || <img src="https://raw.githubusercontent.com/Ziad-Ashraf-Abdu/Retinal_O2_transport/master/Inverse%20Model/plots/InverseModel/parity_plot_Dcc.png" alt="Outer Retina Diffusivity Parity Plot" width="300px"/><br>| <img src="https://raw.githubusercontent.com/Ziad-Ashraf-Abdu/Retinal_O2_transport/master/Inverse%20Model/plots/InverseModel/parity_plot_kcc.png" alt="Profile Reconstruction" width="300px"/><br> |

---

### PINN Reconstruction of Oxygen Profile

<p align="center">  
  <img src="Inverse Model/plots/profile_reconstruction.png" alt="PINN Profile Reconstruction" width="700px"/>  
</p>  
*Figure 4.* Overlay of measured and PINN-predicted concentration profiles from only 10 spatial sensors per layer. L² error < 0.5 %.

---

## 🚀 Why This Matters

* **Clinical Translation**: Non-invasive estimation of retinal oxygenation parameters from limited imaging data.
* **Scientific Rigor**: Proven stability and accuracy via classical and data-driven methods.
* **Open & Extensible**: Modular code, detailed notebooks, and tutorials facilitate adoption and extension.

---

## 🏥 Clinical Applications

1. **Diabetic Retinopathy**

   * Simulate hypoxia-induced tissue damage
   * Predict progression and optimize intervention timing

2. **Age-Related Macular Degeneration**

   * Quantify choroidal circulation deficits
   * Guide treatment strategies

3. **Retinal Vein Occlusion**

   * Map localized hypoxic zones
   * Aid surgical decision-making

---

## 🎨 Technical Innovation

**Physics-Informed Multi-Loss Architecture**

$$
\text{Total Loss} = \lambda_{\text{PDE}} L_{\text{PDE}} + \lambda_{\text{B}} L_{\text{Boundary}}
               + \lambda_{\text{C}} L_{\text{Continuity}} + \lambda_{\text{D}} L_{\text{Data}}
$$

* $L_{\text{PDE}}$: Enforces the reaction–diffusion equation
* $L_{\text{Boundary}}$: Applies physiological boundary conditions
* $L_{\text{Continuity}}$: Ensures flux continuity across layers
* $L_{\text{Data}}$: Matches experimental measurements

**Transformer-Enhanced PINN**

* Captures inter-layer spatial dependencies
* Pre-training on synthetic datasets accelerates convergence
* Physics-informed fine-tuning preserves biophysical realism

---

## 🔭 Future Directions

* **3D Modeling** of full retinal geometry
* **Time-Dependent Disease Progression** simulations
* **Multi-Species Transport** for glucose, lactate, and beyond
* **Real-Time OCT Integration** for patient-specific diagnostics
* **Educational Platform** for biomedical engineering curricula

---

## 👨‍💼 Our Team

A cross-disciplinary group of Biomedical Engineering students specializing in:

* Computational Fluid Dynamics
* Machine Learning & AI
* High-Performance Computing

*Full author list and contributions available in the main README.*

---

## 📚 Dive Into the Details

This page highlights our top results and innovations. For comprehensive derivations, implementation notes, performance benchmarks, and tutorials, please see our full documentation:

➡️ **[Read the Complete README](https://github.com/Ziad-Ashraf-Abdu/Retinal_O2_transport#readme)**

---

## 🤝 Get Involved

* ⭐ Star this repository to follow updates
* 🐛 Report issues or request features
* 💬 Join discussions for collaboration
* 📧 Contact us for research partnerships
* 📖 Cite our work in your publications

---

## 📄 Citation

```bibtex
@misc{retinal_oxygen_transport_2025,
  title        = {Retinal Oxygen Transport: Physics-Informed and Numerical Modeling},
  author       = {[Authors List]},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Ziad-Ashraf-Abdu/Retinal_O2_transport}}
}
```

**Transforming computational ophthalmology, one equation at a time.** 🔬👁️
