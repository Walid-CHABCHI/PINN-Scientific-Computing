# L-BFGS Optimization

Two-phase training: Adam (2000 epochs) followed by L-BFGS (`torch.optim.LBFGS`, `line_search_fn="strong_wolfe"`), applied to both vanilla and hard PINNs for the 1D heat equation.

L-BFGS leverages local curvature (quasi-Newton) where Adam stagnates: the closure function re-evaluates the loss and gradient at each step attempt of the Wolfe line search.

## Results

| Config | Final Loss | Max Error |
|---|---|---|
| Vanilla Adam (12000 epochs) | 1.45e-03 | — |
| Vanilla Adam → L-BFGS | 2.50e-05 | 0.0036 |
| Hard Adam → L-BFGS | **3.17e-08** | **0.0003** |

- `vanilla_adam_lbfgs.ipynb`, `hard_adam_lbfgs.ipynb` — notebooks
- `loss_*.png`, `solution_*.png` — loss curves and solutions
