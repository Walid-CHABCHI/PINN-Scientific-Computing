# pinnDE Experiments — 1D Heat Equation

First experiments using the [pinnDE](https://pinnde.readthedocs.io/) library (TensorFlow).

## Notebooks

- [pinnde_heat_forward.ipynb](file:///Users/mohamedwalidchabchi/stage/00_pinnde_experiments/pinnde_heat_forward.ipynb): Forward problem (1+1D) with a Gaussian initial condition and homogeneous Dirichlet boundary conditions. The code compares the PINN prediction with the analytical solution (maximum absolute error ≈ 0.014).
- [pinnde_heat_inverse.ipynb](file:///Users/mohamedwalidchabchi/stage/00_pinnde_experiments/pinnde_heat_inverse.ipynb): Inverse problem aimed at identifying the diffusivity coefficient $\alpha = 0.08$ from 400 noisy synthetic temperature observations.

## Generated Figures

- [figures/solution_comparison_forward.png](file:///Users/mohamedwalidchabchi/stage/00_pinnde_experiments/figures/solution_comparison_forward.png): Visual comparison of the exact solution, PINN prediction, and absolute error for the forward problem.
- [figures/PDE-solution-pred.png](file:///Users/mohamedwalidchabchi/stage/00_pinnde_experiments/figures/PDE-solution-pred.png): Predicted temperature profile from the inverse model.
- [figures/PDE-epoch-loss.png](file:///Users/mohamedwalidchabchi/stage/00_pinnde_experiments/figures/PDE-epoch-loss.png): Training loss history (residual and data losses) for the inverse model.

