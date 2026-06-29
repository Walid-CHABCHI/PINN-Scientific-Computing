# PINN from scratch — 1D Heat Equation

Implementation of a PINN in pure PyTorch, without any specialized libraries.

## Problem

$$\partial_t u = \alpha \, \partial_{xx} u, \quad \alpha = 5\times 10^{-4}$$

- Initial condition: $u(0, x) = \sin(2\pi x)$
- Boundary conditions: $u(t, 0) = u(t, 1) = 0$ (Dirichlet)

## Method

- MLP 2 → 50 → 50 → 50 → 1, tanh activation
- Derivatives $u_t$, $u_{xx}$ obtained via automatic differentiation (`torch.autograd.grad`)
- Total weighted loss: $300\,\mathcal{L}_{IV} + 100\,\mathcal{L}_{BC} + \mathcal{L}_{PDE}$
- Adam optimizer (lr = 0.001), 12,000 epochs, 12,000 collocation points

## Usage


Adaptation du bruit , Ablation pour la robustesse du bruit.
```bash
python main.py
```
