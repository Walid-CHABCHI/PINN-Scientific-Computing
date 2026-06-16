# PINN Scientific Computing

Solving partial differential equations and inverse problems using **Physics-Informed Neural Networks (PINNs)**, as part of an 8-week internship. The final goal is to solve inverse problems on integro-differential equations (RFE). All implementations are built from scratch in PyTorch.

## Project Structure

| Folder | Content |
|---|---|
| `00_pinnde_experiments/` | First experiments using the [pinnDE](https://pinnde.readthedocs.io/) library: 1D heat equation for both forward and inverse problems (identifying diffusivity α) |
| `01_pinn_from_scratch/` | PINN implemented in pure PyTorch: 1D heat equation, Adam optimizer, weighted loss (physics + initial + boundary conditions) |
| `02_hard_constraints/` | Hard-constrained PINN: initial and boundary conditions enforced by construction ($u = A + B \cdot N$), comparison with the vanilla PINN |
| `03_lbfgs_optimization/` | Two-phase training using Adam followed by L-BFGS (`torch.optim.LBFGS` with strong Wolfe line search) on both vanilla and hard versions |
| `04_inverse_heat/` | Inverse problem for the heat equation: identifying diffusivity α from observations, in vanilla and hard formulations |
| `05_integro_differential/` | Integro-differential equations, RFE inverse problem — core of the internship *(ongoing)* |

## Equation Studied

1D Heat Equation:

$$\partial_t u = \alpha \, \partial_{xx} u, \quad x \in [0,1],\ t \in [0,1]$$

with homogeneous Dirichlet conditions and sinusoidal or Gaussian initial conditions.

## Environment

Python 3.11 — PyTorch, TensorFlow/pinnDE, NumPy, Matplotlib.
