# Hard Constraints

Hard-constrained PINN for the 1D heat equation: initial and boundary conditions are enforced **by construction** via the transformation:

$$u(t,x) = A(t,x) + B(t,x) \cdot N(t,x)$$

where $A = \sin(2\pi x)$ (satisfies initial and boundary conditions) and $B = t \, x (1-x)$ (vanishes exactly where conditions are imposed). This leaves only one loss to minimize: the PDE residual.

## Results

| | Vanilla (01) | Hard |
|---|---|---|
| Losses to tune | 3 (+ weights 300/100) | 1 only |
| IC/BC Conditions | approximated | exact by construction |
| Max error vs exact solution | ~0.014 | **0.0002** |

- `hard_pinn.ipynb` — full implementation and comparison
- `figures/solution_hard_pinn.png`, `figures/loss_hard_pinn.png` — figures
