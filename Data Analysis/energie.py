import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Percorso al dataset (senza "LPSO")
dataset_path = "/Users/lorenzo/Desktop/dataset"

# Percorso di output (senza "Grafici")
grafici_path = "/Users/lorenzo/Desktop"
os.makedirs(grafici_path, exist_ok=True)

# Lista unica di energie
all_energies = []

# Scansiona tutto il dataset e raccoglie TUTTE le energie
for root, dirs, files in os.walk(dataset_path):
    if "energy.npy" in files:
        try:
            energy_file = os.path.join(root, "energy.npy")
            e = np.load(energy_file)
            all_energies.extend(e)
        except Exception as e:
            print(f"Errore in {energy_file}: {e}")

# Se abbiamo trovato delle energie
if all_energies:
    all_energies = np.array(all_energies)

    # 🎨 Plot KDE con area colorata
    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        all_energies,
        bw_adjust=0.3,    # controllo della larghezza della KDE
        fill=True,        # area colorata
        color="steelblue",
        alpha=0.9,        # riempimento più pieno
        linewidth=1       # contorno più sottile
    )

    # ❌ Nessun titolo
    # plt.title("Distribuzione Unica delle Energie (KDE)", fontsize=16)

    # ✅ Asse x con etichetta e numeri più grandi
    plt.xlabel("Energia (eV)", fontsize=14)
    plt.xticks(fontsize=12)

    # ❌ Asse y senza scritte
    plt.ylabel("")
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    n_ticks = 5
    ax.set_yticks(np.linspace(ymin, ymax, n_ticks))
    ax.set_yticklabels([])

    # ✅ Griglia verticale e orizzontale
    plt.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.6)

    # ❌ Nessuna legenda
    plt.legend([], [], frameon=False)

    # 📸 Salva il grafico nella cartella Desktop
    output_path = os.path.join(grafici_path, "distribuzione_unica_energie_kde.png")
    plt.savefig(output_path, dpi=300)
    print(f"✅ Grafico unico (KDE) salvato in: {output_path}")

    plt.show()
else:
    print("❌ Nessun file energy.npy trovato nel dataset.")

