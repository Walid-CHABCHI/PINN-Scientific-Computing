# Problème inverse — équation de la chaleur

Identification de la diffusivité α à partir d'observations de la solution, en PyTorch pur : α devient un paramètre entraînable optimisé conjointement avec les poids du réseau, et une loss de données (MSE sur des mesures synthétiques) s'ajoute au résidu de la PDE.

Deux variantes :
- **vanilla** — loss = IV + BC + CLP(α) + data
- **hard** — conditions exactes par construction, loss = CLP(α) + data

*(en cours)*
