# Physics-Informed Neural Networks for Beam Deflection Analysis

This repository provides the implementation of **Physics-Informed Neural Networks (PINNs)** developed for solving **Euler–Bernoulli beam deflection problems** under various boundary and loading conditions.

The code accompanies the research article:

> **"Physics-Informed Neural Networks for Beam Deflection Analysis: Methodology and Applications"**  
> *Kiarash Baharane et al., 2025.*

---

## 🧩 Overview

This project implements a PINN framework using **DeepXDE** to solve the Euler–Bernoulli beam equation under three configurations:

1. **Cantilever Beam** (Uniformly distributed load)  
2. **Fully Restrained Beam** (Uniformly distributed load)  
3. **Fully Restrained Beam with Mid-Span Point Load**

Each configuration is modeled through the Euler–Bernoulli ODE with adaptive boundary weighting and compared against analytical solutions.

---

## 📁 Repository Structure

