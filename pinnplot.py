import numpy as np
import torch
import matplotlib.pyplot as plt


def evaluer_modele(model, t_range=(0.0, 1.0), x_range=(0.0, 1.0), n=200, device=None):
    t_vals = np.linspace(t_range[0], t_range[1], n)
    x_vals = np.linspace(x_range[0], x_range[1], n)
    X, T = np.meshgrid(x_vals, t_vals)

    entrees = torch.tensor(np.stack([T.ravel(), X.ravel()], axis=1), dtype=torch.float32)
    if device is None:
        device = next(model.parameters()).device
    entrees = entrees.to(device)

    with torch.no_grad():
        U = model(entrees).cpu().numpy().reshape(T.shape)
    return T, X, U


def plot_solution(model, u_exact=None, t_range=(0.0, 1.0), x_range=(0.0, 1.0),
                  n=200, device=None, cmap="jet", save=None):
    T, X, U_pinn = evaluer_modele(model, t_range, x_range, n, device)
    extent = [t_range[0], t_range[1], x_range[0], x_range[1]]

    if u_exact is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(U_pinn.T, cmap=cmap, aspect="auto", origin="lower", extent=extent)
        ax.set_title("Neural network solution")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        if save:
            fig.savefig(save, dpi=150)
        plt.show()
        return U_pinn

    U_exact = np.asarray(u_exact(T, X))
    vmin = min(U_pinn.min(), U_exact.min())
    vmax = max(U_pinn.max(), U_exact.max())
    kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax,
                  aspect="auto", origin="lower", extent=extent)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(U_pinn.T, **kwargs)
    axes[0].set_title("Neural network solution")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(U_exact.T, **kwargs)
    axes[1].set_title("Exact solution")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("x")
    plt.colorbar(im1, ax=axes[1])

    erreur = np.abs(U_pinn - U_exact)
    im2 = axes[2].imshow(erreur.T, cmap="Reds", aspect="auto",
                         origin="lower", extent=extent)
    axes[2].set_title(f"Erreur max = {erreur.max():.4f}")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("x")
    plt.colorbar(im2, ax=axes[2])
    if save:
        fig.savefig(save, dpi=150)
    plt.show()
    return U_pinn, U_exact
