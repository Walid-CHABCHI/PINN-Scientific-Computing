# Expériences pinnDE — équation de la chaleur 1D

Premières expériences avec la librairie [pinnDE](https://pinnde.readthedocs.io/) (TensorFlow).

- `pinnde_heat_forward.ipynb` : problème direct (1+1D), condition initiale gaussienne, comparaison avec la solution analytique (erreur max ≈ 0.014)
- `pinnde_heat_inverse.ipynb` : problème inverse — identification de la diffusivité α = 0.08 à partir de 400 points de données bruitées
- `PDE-solution-pred.png`, `PDE-epoch-loss.png` : figures générées par les plotters pinnDE
