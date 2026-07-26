import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Percorso al dataset
dataset_path = "/Users/lorenzo/Desktop/dataset"

# Percorso alla cartella per salvare il grafico
grafici_path = "/Users/lorenzo/Desktop"
os.makedirs(grafici_path, exist_ok=True)

# Dizionario per salvare le norme delle forze per tipo di simulazione
force_norms_by_type = defaultdict(list)

file_count = 0

# Scorri tutte le sottocartelle e raccogli i dati da force.npy
for root, dirs, files in os.walk(dataset_path):
    if "force.npy" in files:
        try:
            force_file = os.path.join(root, "force.npy")
            forces = np.load(force_file)

            file_count += 1

            if forces.ndim == 3:
                norms = np.linalg.norm(forces, axis=2).flatten()
            elif forces.ndim == 2:
                norms = np.linalg.norm(forces, axis=1)
            else:
                raise ValueError("Forma non riconosciuta del file force.npy")

            rel_path = os.path.relpath(root, dataset_path)
            sim_type = rel_path.split(os.sep)[0] if rel_path != "." else "root"

            force_norms_by_type[sim_type].extend(norms.tolist())

        except Exception as e:
            print(f"Errore in {force_file}: {e}")

# Pulizia dei dati
for sim_type in sorted(force_norms_by_type.keys()):
    arr = np.array(force_norms_by_type[sim_type], dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    arr = arr[np.isfinite(arr)]
    force_norms_by_type[sim_type] = arr

# 🎨 Plot
plt.figure(figsize=(12, 7))

types = sorted(force_norms_by_type.keys())
palette = sns.color_palette(n_colors=max(1, len(types)))

for i, sim_type in enumerate(types):
    data = force_norms_by_type[sim_type]
    if data.size == 0:
        continue

    unique_vals = np.unique(data)
    if unique_vals.size < 2 or data.size < 5:
        sns.histplot(
            data,
            bins=min(10, max(1, data.size)),
            stat='density',
            alpha=0.6,
            label=sim_type,
            color=palette[i]
        )
        if unique_vals.size == 1:
            plt.axvline(unique_vals[0], color=palette[i], linestyle='--', linewidth=1)
    else:
        sns.kdeplot(
            data,
            bw_adjust=0.3,
            fill=True,
            color=palette[i],
            alpha=0.6,
            label=sim_type
        )

# ❌ Nessun titolo
plt.xlabel("Modulo della forza |F| (eV/Å)")

# ❌ Nessuna scala/etichetta sull'asse verticale
plt.ylabel("")

# Prendi l'asse corrente e imposta tick verticali proporzionati automaticamente
ax = plt.gca()
ymin, ymax = ax.get_ylim()
n_ticks = 5
ax.set_yticks(np.linspace(ymin, ymax, n_ticks))
ax.set_yticklabels([])  # etichette invisibili

# ✅ Griglia verticale e orizzontale
plt.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.7)

plt.legend(loc="upper right")
plt.xlim(0, 15.0)

# Salva il grafico
output_path = os.path.join(grafici_path, "distribuzione_forze_kde.png")
plt.savefig(output_path, dpi=300)
print(f"✅ Grafico salvato in: {output_path}")

plt.show()

