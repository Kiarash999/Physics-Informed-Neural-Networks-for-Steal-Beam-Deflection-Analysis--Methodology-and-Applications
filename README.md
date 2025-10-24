# Physics-Informed Neural Networks for Beam Deflection Analysis

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/DeepXDE--TensorFlow-orange)

This repository contains the code and experiments for the paper:

**"Physics-Informed Neural Networks for Beam Deflection Analysis: Methodology and Applications"**  
*Kiarash Baharan et al., 2025.*

The implementation uses **DeepXDE** + **TensorFlow** to solve Euler–Bernoulli beam deflection problems under several boundary and loading conditions.

---

## Overview

This project implements a PINN framework using **DeepXDE** to solve the Euler–Bernoulli beam equation under three configurations:

1. **Cantilever Beam** (Uniformly distributed load)  
2. **Fully Restrained Beam** (Uniformly distributed load)  
3. **Fully Restrained Beam with Mid-Span Point Load**

Each configuration is modeled through the Euler–Bernoulli ODE with adaptive boundary weighting and compared against analytical solutions.

---

## Repository structure

```
.
├── LICENSE
├── README.md
├── requirements.txt
│
├── config.py             # Global parameters (geometry, material, seeds, paths)
├── train.py              # Main entry point for training (Adam → L-BFGS)
├── models.py             # Network architectures (FNN builder)
├── bcs.py                # Geometry, boundary conditions, PDE data constructors
├── ode.py                # Differential equations (Euler–Bernoulli, Gaussian load)
├── analy.py              # Analytical/reference solutions
├── utils.py              # Plotting and helper functions
│
├── figures/              # Generated plots (loss, deflection curves)
├── results/              # Saved models, logs, checkpoints
├── examples/             # Optional example configs or notebooks
│
├── .gitignore
└── CITATION.cff          # (optional) Citation metadata for GitHub
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
numpy
scipy
matplotlib
tensorflow>=2.10
deepxde>=1.9
tqdm
```

---

## Usage

```bash
# Cantilever beam (default)
python train.py --case cantilever

# Fixed-fixed beam
python train.py --case fixed

# Fixed-fixed beam with mid-span point load
python train.py --case point
```

The script automatically builds the PDE data, constructs the FNN, trains the PINN model, and generates figures under `figures/`.

---

## Results

Results are saved to `figures/` and `results/`.  
Typical outputs include:

- **solution\_<case>.png** — predicted vs analytical deflection  
- **loss\_<case>.png** — loss convergence history  
- Reported relative \( L_2 \) errors in the order of 1e-4 – 1e-3.

---

## Citation

If you use this repository in your research, please cite:

```
@article{baharan2025pinnbeam,
  title={Physics-Informed Neural Networks for Beam Deflection Analysis: Methodology and Applications},
  author={Baharan, Kiarash and Dr. hassan Mirzabozorg},
  year={2025},
  journal={Advanced Computational Engineering}
}
```

---

## License

This project is licensed under the **MIT License** — see `LICENSE` file.

---

## 👤 Contact

Kiarash Baharan  
Email: [kiarashbaharan@gmail.com]  
GitHub: [github.com/kiarash999]
