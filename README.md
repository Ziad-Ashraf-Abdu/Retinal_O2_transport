# Retinal Oxygen Transport via Physics-Informed Neural Networks (PINNs)

This repository implements a Physics-Informed Neural Network (PINN) model to estimate physiologically meaningful parameters governing oxygen diffusion and consumption in the retina. The goal is to reconstruct hidden tissue characteristics such as diffusion coefficients and metabolic consumption rates from spatial oxygen concentration profiles.

> ⚠️ Currently, this version focuses on two retinal layers. Support for all four retinal layers (choroid, outer retina, inner retina, vitreous) will be added soon. The framework is fully extensible.

## 🔬 Background & Reference

This project is inspired by the classical physiological modeling described in the publication:

> "Retinal Oxygen Transport: A Mathematical Model" – (Refer to `4 Retinal O2 transport.pdf` in the repo)

The PDE governing steady-state oxygen transport in each layer is a 1D diffusion-reaction equation:

$$
D_i \frac{d^2 C_i}{dz^2} - k_i C_i = 0
$$

Where:

* $D_i$: Diffusion coefficient for layer $i$
* $k_i$: Consumption rate in layer $i$
* $C_i(z)$: Oxygen concentration at depth $z$

## 📌 Problem Statement

This project implements an inverse PINN that:

* Accepts spatial oxygen concentration profiles as input (e.g. from sensors or synthetic simulations)
* Predicts the underlying biophysical parameters: diffusion coefficients (D2, D3), metabolic rates (k2, k3), and boundary concentrations (C0, CL)

## 🧠 Why PINNs?

Unlike traditional black-box models, Physics-Informed Neural Networks:

* Embed PDEs directly into the training loss
* Require less data by leveraging known physical laws
* Provide interpretable parameters that satisfy physiological constraints

We use `pinnstorch`, a PyTorch Lightning-compatible PINN library.

Inspired by:

* [https://github.com/rezaakb/pinns-torch.git](https://github.com/rezaakb/pinns-torch.git)

## 📂 Repository Structure

```
├── O2_profile.py           # Main training & evaluation script
├── requirements.txt        # All Python dependencies
├── 4 Retinal O2 transport.pdf  # Foundational reference for PDE model
├── InnerRetina.pdf         # Additional validation literature
├── enhanced_profiles/      # Directory storing generated data
├── enhanced_checkpoints/   # Folder for saved models
├── output images/          # Parity plots, reconstructions, error visualizations
```

## 🔍 Features

* Residual-based PINN training with boundary constraints
* Custom network architecture with attention and multi-head output
* Robust parameter normalization & physical constraints
* Enhanced visualization for parity plots and reconstructed profiles
* Support for multiple training runs and early stopping

## 📊 Parameters Estimated

* D2: Diffusion coefficient for second layer
* D3: Diffusion coefficient for third layer
* k2: Metabolic rate in second layer
* k3: Metabolic rate in third layer
* C0: Oxygen concentration at z = 0
* CL: Oxygen concentration at z = L

➡️ Note: Upcoming versions will include D1, k1 (outer retina) and D4, k4 (vitreous)

## 🚀 How to Run

### 🟢 Recommended: Google Colab

PINNs are computationally demanding. For ease of use and free GPU support:

1. Visit Google Colab: [https://colab.research.google.com](https://colab.research.google.com)
2. Upload `O2_profile.py`
3. Make sure the runtime type is set to `GPU`
4. Install requirements using:

```bash
!pip install -r requirements.txt
```

5. Run the script and monitor training curves.

Colab Tip:

* If you encounter timeout errors, reduce the number of profiles or training epochs in the config section.

---

### 🖥️ Local Run (Advanced)

Make sure Python 3.9+ and PyTorch (with CUDA support) are installed.

1. Clone the repo:

```bash
git clone https://github.com/Ziad-Ashraf-Abdu/Retinal_O2_transport.git
cd Retinal_O2_transport
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run:

```bash
python O2_profile.py
```

Training will start and save checkpoints in `enhanced_checkpoints/`.

## 📈 Outputs

* `enhanced_parity_plots.png`: Predicted vs True values for each parameter
* `enhanced_profile_reconstructions.png`: Visual reconstruction of O2 profiles
* `enhanced_error_analysis.png`: Error trends, boxplots, and correlation

## 🧪 Sample Results

Initial results show the model is capable of estimating parameters with good relative accuracy. See the visualizations for more insights.

Average error after tuning (target < 10%):

* D2: \~6%
* k2: \~8%
* D3: \~5%
* k3: \~7%
* C0/CL: \~4%

## 🧑‍🤝‍🧑 Authors

This project was developed as a collaborative research project by:

* Zeyad Ashraf \[github.com/Ziad-Ashraf-Abdu]
* \[Teammate 1] – Focused on Inner Retina PDE formulation
* \[Teammate 2] – Validation, PINNs integration, evaluation

Feel free to open issues or contact us for questions and suggestions.

---

## 📚 References

1. \[Main PDE Reference] 4 Retinal O2 Transport.pdf
2. pinnstorch GitHub: [https://github.com/rezaakb/pinns-torch.git](https://github.com/rezaakb/pinns-torch.git)
3. Physics-Informed Neural Networks (Raissi et al. 2019)

---

## 🌟 Future Plans

* Add support for all 4 layers
* Integrate real patient data (if available)
* Enable time-dependent modeling (non-steady-state)
* Extend to 2D/3D retinal slices

---

Made with 🧠 + ❤️ for computational physiology and medical AI.
