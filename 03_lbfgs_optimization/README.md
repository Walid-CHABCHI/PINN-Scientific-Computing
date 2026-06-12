# Optimisation L-BFGS

Entraînement en deux phases : Adam (2000 époques) puis L-BFGS (`torch.optim.LBFGS`, `line_search_fn="strong_wolfe"`), appliqué au PINN vanilla et au PINN hard sur l'équation de la chaleur 1D.

Le L-BFGS exploite la courbure locale (quasi-Newton) là où Adam stagne : la closure réévalue loss et gradient à chaque essai de pas de la recherche linéaire de Wolfe.

## Résultats

| Config | Loss finale | Erreur max |
|---|---|---|
| Vanilla Adam (12000 ép.) | 1.45e-03 | — |
| Vanilla Adam → L-BFGS | 2.50e-05 | 0.0036 |
| Hard Adam → L-BFGS | **3.17e-08** | **0.0003** |

- `vanilla_adam_lbfgs.ipynb`, `hard_adam_lbfgs.ipynb` — notebooks
- `loss_*.png`, `solution_*.png` — courbes de loss et solutions
