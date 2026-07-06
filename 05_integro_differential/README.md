# Équations Intégro-Différentielles — Problème Inverse RFE

Ce dossier contient les travaux sur la résolution des équations de transfert radiatif (RTE/RFE) à l'aide de PINNs.

## Cas Étudiés

### [Cas 1 (1D)] (pinn_integro_pytorch.ipynb)
Résolution de l'Équation du Transfert Radiatif en 1D (espace + direction) avec conditions aux limites entrantes asymétriques.
*   **Vanilla PINN** : [pinn_integro_pytorch.ipynb](file:///Users/mohamedwalidchabchi/stage/05_integro_differential/pinn_integro_pytorch.ipynb) (Adam + L-BFGS)
*   **Hard Constraints PINN** : [hard_pinn_integro_pytorch.ipynb](file:///Users/mohamedwalidchabchi/stage/05_integro_differential/hard_pinn_integro_pytorch.ipynb) (Adam + L-BFGS)

### [Cas 2 (2D)] (pinn_integro_pytorch_cas2.ipynb)
Extension de l'Équation du Transfert Radiatif en 2D (espace 2D + angles 3D $\theta, \phi$), configuration du benchmark de **Crosbie et Schrenker (1984)** : carré purement diffusant ($\omega=1$), paroi haute exposée à un rayonnement diffus $I=1$.
*   **Vanilla PINN** : [pinn_integro_pytorch_cas2.ipynb](file:///Users/mohamedwalidchabchi/stage/05_integro_differential/pinn_integro_pytorch_cas2.ipynb) (Adam + L-BFGS)
*   **Vanilla VPINN** : [vpinn_integro_pytorch_cas2.ipynb](file:///Users/mohamedwalidchabchi/stage/05_integro_differential/vpinn_integro_pytorch_cas2.ipynb) — formulation faible (base test sinus $K \times K$, intégration par parties, zéro autograd dans la loss physique), à comparer au PINN fort à budget égal.
*   **Validation** : solveur DOM 2D intégré aux notebooks (upwind + itération de la source) + ancre analytique exacte $G(0.5, 0.5) = \pi$ (argument de superposition des 4 rotations du carré, valable pour toute épaisseur optique).
*   ⚠️ Les anciennes valeurs de référence codées en dur (attribuées à tort aux tables de C&S, en réalité générées par une IA) étaient fausses — elles impliquaient $G(0.5,0.5) = 4.06 \neq \pi$. Le PINN prédisait déjà la bonne solution.





Entrainer mon modele sur l'integral , et voir les pinns variationnels.
Exprimer lequation en radial.