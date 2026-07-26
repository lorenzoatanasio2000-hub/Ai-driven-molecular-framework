import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import defaultdict, Counter

# === CONFIG ===
dataset_root = "/Users/lorenzo/Desktop/LPSO/dataset"
output_path = "/Users/lorenzo/Desktop/Grafici"
os.makedirs(output_path, exist_ok=True)

def load_types(struct_path):
    """Carica type.raw e type_map.raw"""
    try:
        types = np.loadtxt(os.path.join(struct_path, "type.raw"), dtype=int)
        with open(os.path.join(struct_path, "type_map.raw")) as f:
            type_map = [line.strip() for line in f.readlines()]
        symbols = [type_map[t] for t in types]
        return types, type_map, symbols
    except:
        return None, None, None

def find_structures(root):
    """Trova tutte le cartelle con type.raw e type_map.raw"""
    structures = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "type.raw" in filenames and "type_map.raw" in filenames:
            structures.append(dirpath)
    return structures

def load_coordinates(struct_path, types):
    """Carica tutti i frame coerenti da coord.npy"""
    frames = []
    for sub in os.listdir(struct_path):
        if not sub.startswith("set."):
            continue
        coord_path = os.path.join(struct_path, sub, "coord.npy")
        if not os.path.exists(coord_path):
            continue
        try:
            coords_raw = np.load(coord_path)
            if coords_raw.ndim != 2 or coords_raw.shape[1] % 3 != 0:
                continue

            n_atoms = coords_raw.shape[1] // 3
            coords = coords_raw.reshape(-1, n_atoms, 3)

            if len(types) != n_atoms:
                if n_atoms % len(types) == 0:
                    repeat = n_atoms // len(types)
                    types_expanded = np.tile(types, repeat)
                else:
                    continue

            for frame in coords:
                frames.append(frame.flatten())
        except:
            continue
    return frames

def composition_to_string(symbols):
    """Trasforma una lista di simboli in una formula compatta, es. ['Li','Li','P'] -> 'Li2P1'"""
    counter = Counter(symbols)
    comp_str = ''.join(f"{el}{counter[el]}" for el in sorted(counter))
    return comp_str

# === MAIN ===
all_structures = find_structures(dataset_root)
print(f"Trovate {len(all_structures)} strutture candidate.")

composition_dict = defaultdict(list)

# Raccogli tutti i frame per composizione chimica
for struct in all_structures:
    types, type_map, symbols = load_types(struct)
    if types is None:
        continue

    frames = load_coordinates(struct, types)
    if frames:
        comp_str = composition_to_string(symbols)
        composition_dict[comp_str].extend(frames)

# PCA per ogni composizione chimica
for comp_str, frames in composition_dict.items():
    X = np.array(frames)
    if len(X) < 2:
        continue  # serve almeno 2 punti per PCA

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], s=15, alpha=0.8, c="steelblue")
    plt.title(f"PCA - Composizione {comp_str}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)

    output_file = os.path.join(output_path, f"PCA_{comp_str}.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"✅ Grafico salvato: {output_file}")

