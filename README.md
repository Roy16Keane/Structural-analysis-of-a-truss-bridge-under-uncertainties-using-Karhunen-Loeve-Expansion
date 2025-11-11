# Structural-analysis-of-a-truss-bridge-under-uncertainties-using-Karhunen-Loeve-Expansion
# Structural Analysis of a Truss Bridge with Uncertainties using Karhunen–Loève Expansion

This project, originally completed as part of my MSc thesis at the **University of Glasgow**, explores **uncertainty quantification (UQ)** for structural systems using the **Karhunen–Loève (KL) expansion**, a spectral method for representing random fields.

---

## Objective

To benchmark and compare several methods for quantifying structural response uncertainty under stochastic loading conditions:

1. **Analytical formulation** (statFEM inspired)
2. **Monte Carlo sampling**
3. **Spectral method (KL expansion)**
4. **Neural network surrogate model**

---

## Problem Definition

- Structure: 2D Warren truss bridge (7 nodes, 11 elements)
- Material: HEA 300 steel (E = 200 GPa)
- Load: Random normal distribution  
  - Mean = 40 tons  
  - Std. dev. = 10% (nodes 3 and 5)

---

## Methodology

| Approach | Description | Pros | Cons |
|-----------|--------------|------|------|
| **Monte Carlo** | Random sampling propagation | Simple, robust | High computational cost |
| **Spectral (KL)** | Decomposition of random field using eigenmodes | Efficient, compact | Accuracy depends on #modes |
| **Neural Network** | Learn nonlinear mapping f → u | Fast inference | Requires many data, less interpretable |

---

## Results Summary

- Minimal covariance difference across all methods  
- **KL expansion** achieved comparable accuracy to Monte Carlo at **lower computation cost**
- Neural Network surrogate required more training time but generalized well to unseen loads

---



