# Équations Intégro-Différentielles — Problème Inverse RFE

Ce dossier contient les travaux sur la résolution des équations de transfert radiatif (RTE/RFE) à l'aide de PINNs.

## Cas Étudiés

### [Cas 1 (1D)] (pinn_integro_pytorch.ipynb)
Résolution de l'Équation du Transfert Radiatif en 1D (espace + direction) avec conditions aux limites entrantes asymétriques.
*   **Vanilla PINN** : [pinn_integro_pytorch.ipynb](file:///Users/mohamedwalidchabchi/stage/05_integro_differential/pinn_integro_pytorch.ipynb) (Adam + L-BFGS)
*   **Hard Constraints PINN** : [hard_pinn_integro_pytorch.ipynb](file:///Users/mohamedwalidchabchi/stage/05_integro_differential/hard_pinn_integro_pytorch.ipynb) (Adam + L-BFGS)

### [Cas 2 (2D)] (pinn_integro_pytorch_cas2.ipynb)
Extension de l'Équation du Transfert Radiatif en 2D (espace 2D + angles 3D $\theta, \phi$) d'après le benchmark classique de **Crosbie et Schrenker [30]**.
*   **Vanilla PINN** : [pinn_integro_pytorch_cas2.ipynb](file:///Users/mohamedwalidchabchi/stage/05_integro_differential/pinn_integro_pytorch_cas2.ipynb) (Adam + L-BFGS, *structure prête*)

