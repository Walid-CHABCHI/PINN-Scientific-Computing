# PINN Scientific Computing

Résolution d'équations aux dérivées partielles et de problèmes inverses par **Physics-Informed Neural Networks (PINNs)**, dans le cadre d'un stage de 8 semaines. L'objectif final est la résolution de problèmes inverses sur des équations intégro-différentielles (RFE). Toutes les implémentations sont faites from scratch en PyTorch.

## Structure du projet

| Dossier | Contenu |
|---|---|
| `00_pinnde_experiments/` | Premières expériences avec la librairie [pinnDE](https://pinnde.readthedocs.io/) : équation de la chaleur 1D en problème direct et inverse (identification de la diffusivité α) |
| `01_pinn_from_scratch/` | PINN implémenté en PyTorch pur : équation de la chaleur 1D, optimiseur Adam, loss pondérée (physique + conditions initiales + bords) |
| `02_lbfgs_optimization/` | Entraînement avec L-BFGS (`torch.optim.LBFGS`), comparaison avec Adam *(à venir)* |
| `03_hard_constraints/` | Contraintes dures vs PINN vanilla *(à venir)* |
| `04_integro_differential/` | Équations intégro-différentielles, problème inverse RFE — cœur du stage *(à venir)* |
| `cours_scmi/` | TP du cours SciML M2 (interpolation, incertitude, EDP) |

## Équation étudiée

Équation de la chaleur 1D :

$$\partial_t u = \alpha \, \partial_{xx} u, \quad x \in [0,1],\ t \in [0,1]$$

avec conditions de Dirichlet homogènes et condition initiale sinusoïdale ou gaussienne.

## Environnement

Python 3.11 — PyTorch, TensorFlow/pinnDE, NumPy, Matplotlib.
