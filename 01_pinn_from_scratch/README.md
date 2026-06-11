# PINN from scratch — équation de la chaleur 1D

Implémentation d'un PINN en PyTorch pur, sans librairie spécialisée.

## Problème

$$\partial_t u = \alpha \, \partial_{xx} u, \quad \alpha = 5\times 10^{-4}$$

- Condition initiale : $u(0, x) = \sin(2\pi x)$
- Conditions aux bords : $u(t, 0) = u(t, 1) = 0$ (Dirichlet)

## Méthode

- MLP 2 → 50 → 50 → 50 → 1, activation tanh
- Dérivées $u_t$, $u_{xx}$ obtenues par différentiation automatique (`torch.autograd.grad`)
- Loss totale pondérée : $300\,\mathcal{L}_{IV} + 100\,\mathcal{L}_{BC} + \mathcal{L}_{PDE}$
- Optimiseur Adam (lr = 0.001), 12 000 époques, 12 000 points de collocation

## Utilisation

```bash
python main.py
```
