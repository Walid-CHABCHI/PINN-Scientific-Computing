import torch
import torch.nn as nn
import torch.optim as optim

t_min, t_max = 0.0, 1.0
x_min, x_max = 0.0, 1.0


class PinnHeatEq(nn.Module):
    def __init__(self):
        super().__init__()
        self.couche_entree = nn.Linear(2, 50)
        self.couche_cachee1 = nn.Linear(50, 50)
        self.couche_cachee2 = nn.Linear(50, 50)
        self.couche_sortie = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.tanh(self.couche_entree(x))
        x = torch.tanh(self.couche_cachee1(x))
        x = torch.tanh(self.couche_cachee2(x))
        return self.couche_sortie(x)


def generer_points_collocation(n_pde):
    t_colloc = torch.rand(n_pde, 1) * (t_max - t_min) + t_min
    x_colloc = torch.rand(n_pde, 1) * (x_max - x_min) + x_min
    return t_colloc.float(), x_colloc.float()


def generer_points_initiaux(n_iv):
    t_init = torch.zeros(n_iv, 1)
    x_init = torch.rand(n_iv, 1) * (x_max - x_min) + x_min
    return t_init.float(), x_init.float()


def generer_points_bords(n_bords):
    t_bord = torch.rand(n_bords, 1) * (t_max - t_min) + t_min
    x_gauche = torch.ones(n_bords // 2, 1) * x_min
    x_droite = torch.ones(n_bords // 2, 1) * x_max
    x_bord = torch.cat([x_gauche, x_droite], dim=0)
    return t_bord.float(), x_bord.float()


def calc_iv_loss(model, t_init, x_init, u_exact_init):
    u_pred = model(torch.cat([t_init, x_init], dim=1))
    return torch.mean((u_pred - u_exact_init) ** 2)


def calc_bc_loss(model, t_bord, x_bord, u_exact_bord):
    u_pred = model(torch.cat([t_bord, x_bord], dim=1))
    return torch.mean((u_pred - u_exact_bord) ** 2)


def calc_clp_loss(model, t_colloc, x_colloc, alpha):
    t_colloc.requires_grad_(True)
    x_colloc.requires_grad_(True)

    u_pred = model(torch.cat([t_colloc, x_colloc], dim=1))

    u_t = torch.autograd.grad(
        outputs=u_pred,
        inputs=t_colloc,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True,
    )[0]

    u_x = torch.autograd.grad(
        outputs=u_pred,
        inputs=x_colloc,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True,
    )[0]

    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=x_colloc,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
    )[0]

    residu = u_t - alpha * u_xx
    return torch.mean(residu ** 2)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)

    alpha = 0.0005
    epochs = 12000

    modele = PinnHeatEq().to(device)
    optimizer = optim.Adam(modele.parameters(), lr=0.001)

    t_colloc, x_colloc = generer_points_collocation(12000)
    t_init, x_init = generer_points_initiaux(1000)
    t_bord, x_bord = generer_points_bords(1000)

    u_exact_init = torch.sin(2 * torch.pi * x_init)
    u_exact_bord = torch.zeros_like(x_bord)

    t_colloc, x_colloc = t_colloc.to(device), x_colloc.to(device)
    t_init, x_init = t_init.to(device), x_init.to(device)
    t_bord, x_bord = t_bord.to(device), x_bord.to(device)
    u_exact_init = u_exact_init.to(device)
    u_exact_bord = u_exact_bord.to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()

        loss_iv = calc_iv_loss(modele, t_init, x_init, u_exact_init)
        loss_bc = calc_bc_loss(modele, t_bord, x_bord, u_exact_bord)
        loss_clp = calc_clp_loss(modele, t_colloc, x_colloc, alpha)

        loss_totale = 300 * loss_iv + 100 * loss_bc + loss_clp
        loss_totale.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoque {epoch:05d} | "
                  f"Loss totale: {loss_totale.item():.6f} | "
                  f"CLP: {loss_clp.item():.6f} | "
                  f"BC: {loss_bc.item():.6f} | "
                  f"IV: {loss_iv.item():.6f}")

    return modele


if __name__ == "__main__":
    main()
