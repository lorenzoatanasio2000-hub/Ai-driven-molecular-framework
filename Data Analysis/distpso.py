import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# --- CONFIGURAZIONE ---
dataset_path = "/Users/lorenzo/Desktop/LPSO/dataset"
output_path = "/Users/lorenzo/Desktop/Grafici"

atom1 = "P"
atom2_list = ["S", "O"]  # Lista coppie da analizzare

# --- FUNZIONI ---

def load_types(structure_path):
    type_raw_path = os.path.join(structure_path, "type.raw")
    type_map_path = os.path.join(structure_path, "type_map.raw")

    with open(type_raw_path, "r") as f:
        types = np.array([int(x) for x in f.read().split()])

    with open(type_map_path, "r") as f:
        type_map = [line.strip() for line in f.readlines()]

    return types, np.array(type_map)

def get_min_distances(coords, types, type_map, atom2):
    atomic_symbols = type_map[types]
    Li_idx = np.where(atomic_symbols == atom1)[0]
    atom2_idx = np.where(atomic_symbols == atom2)[0]

    min_dists = []
    for i in Li_idx:
        if len(atom2_idx) == 0:
            continue
        dists = np.linalg.norm(coords[i] - coords[atom2_idx], axis=1)
        min_d = np.min(dists)
        min_dists.append(min_d)
    return min_dists

def process_structure(structure_path, atom2):
    print(f"Processo struttura: {structure_path} per legami minimi {atom1}-{atom2}")

    types, type_map = load_types(structure_path)
    all_dists = []

    sets = [d for d in os.listdir(structure_path) if d.startswith("set.")]
    if not sets:
        print(f"Nessuna cartella set.xxx trovata in {structure_path}")
        return all_dists

    for s in sets:
        set_path = os.path.join(structure_path, s)
        npy_files = [f for f in os.listdir(set_path) if f.endswith(".npy") and 'coord' in f.lower()]

        if not npy_files:
            print(f"Nessun file coordinate (.npy con 'coord') in {set_path}")
            continue

        for npy_file in npy_files:
            npy_path = os.path.join(set_path, npy_file)
            coords = np.load(npy_path)

            if coords.ndim != 2:
                print(f"Attenzione: formato coordinate errato in {npy_file} {coords.shape}, salto file.")
                continue

            N_atoms_coord = coords.shape[1] // 3
            if coords.shape[1] % 3 != 0:
                print(f"Formato coordinate non valido in {npy_file}, salto.")
                continue

            N_types = len(types)

            if N_atoms_coord != N_types:
                if N_atoms_coord % N_types == 0:
                    repeat_factor = N_atoms_coord // N_types
                    types_expanded = np.tile(types, repeat_factor)
                else:
                    print(f"Numero atomi coordinate ({N_atoms_coord}) non multiplo di tipi ({N_types}), salto file.")
                    continue
            else:
                types_expanded = types

            coords = coords.reshape(coords.shape[0], N_atoms_coord, 3)

            for frame in coords:
                dists = get_min_distances(frame, types_expanded, type_map, atom2)
                all_dists.extend(dists)

    if len(all_dists) == 0:
        print(f"Nessun legame minimo {atom1}-{atom2} trovato in {structure_path}")

    return all_dists

def find_structures(root):
    structures = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "type.raw" in filenames and "type_map.raw" in filenames:
            structures.append(dirpath)
    return structures

# --- SCRIPT PRINCIPALE ---

if __name__ == "__main__":
    os.makedirs(output_path, exist_ok=True)
    structures = find_structures(dataset_path)
    print(f"Trovate {len(structures)} strutture.")

    all_distances = {atom2: [] for atom2 in atom2_list}

    for struct in structures:
        for atom2 in atom2_list:
            dists = process_structure(struct, atom2)
            all_distances[atom2].extend(dists)

    if all(len(all_distances[atom2]) == 0 for atom2 in atom2_list):
        print("Nessun legame minimo trovato in tutto il dataset.")
    else:
        bins = np.linspace(0, 5, 100)
        plt.figure(figsize=(8,5))

        for atom2 in atom2_list:
            if len(all_distances[atom2]) == 0:
                print(f"Nessun legame minimo {atom1}-{atom2} trovato nel dataset.")
                continue
            hist, bin_edges = np.histogram(all_distances[atom2], bins=bins, density=True)
            hist_smooth = gaussian_filter1d(hist, sigma=2)
            plt.plot(bin_edges[:-1], hist_smooth, label=f'Distribuzione minima {atom1}-{atom2}')

        plt.xlabel("Distanza minima (Å)")
        plt.ylabel("Densità di probabilità")
        plt.title(f"Distribuzione distanza minima {atom1}-X")
        plt.legend()
        plt.grid(True)

        filename = f"Distribuzione_minima_{atom1}_S_O.png"
        plt.savefig(os.path.join(output_path, filename))
        plt.close()
        print(f"Grafico complessivo salvato: {filename}")

