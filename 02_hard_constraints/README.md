# Hard constraints

PINN à contraintes dures sur l'équation de la chaleur 1D : les conditions initiales et aux bords sont imposées **par construction** via la transformation

$$u(t,x) = A(t,x) + B(t,x) \cdot N(t,x)$$

avec $A = \sin(2\pi x)$ (satisfait la condition initiale et les bords) et $B = t \, x (1-x)$ (s'annule exactement là où les conditions sont imposées). Il ne reste qu'une seule loss à minimiser : le résidu de la PDE.

## Résultats

| | Vanilla (01) | Hard |
|---|---|---|
| Losses à régler | 3 (+ poids 300/100) | 1 seule |
| Conditions IC/BC | approchées | exactes par construction |
| Erreur max vs solution exacte | ~0.014 | **0.0002** |

- `hard_pinn.ipynb` — implémentation et comparaison complète
- `solution_hard_pinn.png`, `loss_hard_pinn.png` — figures
