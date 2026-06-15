# Inverse Problem — Heat Equation

Identifying diffusivity α from solution observations in pure PyTorch: α becomes a trainable parameter optimized jointly with the network weights, and a data loss (MSE on synthetic measurements) is added to the PDE residual.

Two variants:
- **vanilla** — loss = IV + BC + CLP(α) + data
- **hard** — exact conditions by construction, loss = CLP(α) + data

*(ongoing)*
