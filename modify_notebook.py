import json

filename = "05_integro_differential/pinn_integro_pytorch.ipynb"

with open(filename, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if "N_quadrature = 32" in line:
                line = line.replace("N_quadrature = 32", "N_quadrature = 16")
            elif "max_iter=1000" in line:
                line = line.replace("max_iter=1000", "max_iter=300")
            elif "max_eval=1250" in line:
                line = line.replace("max_eval=1250", "max_eval=400")
            new_source.append(line)
        cell['source'] = new_source

with open(filename, 'w') as f:
    json.dump(nb, f, indent=1)
print("Notebook modified successfully.")
