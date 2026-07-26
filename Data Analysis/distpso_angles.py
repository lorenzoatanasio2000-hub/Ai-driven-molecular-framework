#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# --- CONFIGURAZIONE ---
dataset_path = "/Users/lorenzo/Desktop/dataset"
output_path = "/Users/lorenzo/Desktop"  # output direttamente sul Desktop

central_atom = "P"
neighbor_atoms = ["S", "O"]
angle_types_to_make = ["S-P-S", "S-P-O", "O-P-O"]
cutoff = 2.6  # Å
bins = np.linspace(0, 180, 181)

# --- FUNZIONI ---

def load_types(structure_path):
    type_raw_path = os.path.join(structure_path, "type.raw")
    type_map_path = os.path.join(structure_path, "type_map.raw")
    with open(type_raw_path, "r") as f:
        types = np.array([int(x) for x in f.read().split()])
    with open(type_map_path, "r") as f:
        type_map = np.array([line.strip() for line in f.readlines()])
    return types, type_map

def compute_angle_deg(center, p1, p2):
    v1 = p1 - center
    v2 = p2 - center
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def process_structure_for_angles(structure_path):
    print(f"Processo struttura: {structure_path}")
    types, type_map = load_types(structure_path)
    angles_dict = {atype: [] for atype in angle_types_to_make}

    sets = [d for d in os.listdir(structure_path) if d.startswith("set.")]
    if not sets:
        print(f"Nessuna cartella set.xxx trovata in {structure_path}")
        return angles_dict

    for s in sets:
        set_path = os.path.join(structure_path, s)
        npy_files = [f for f in os.listdir(set_path) if f.endswith(".npy") and 'coord' in f.lower()]
        for npy_file in npy_files:
            npy_path = os.path.join(set_path, npy_file)
            coords = np.load(npy_path)
            if coords.ndim != 2 or coords.shape[1] % 3 != 0:
                print(f"Formato coordinate non valido in {npy_file}, salto.")
                continue

            N_atoms_coord = coords.shape[1] // 3
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
            atomic_symbols = type_map[types_expanded]

            central_idx = np.where(atomic_symbols == central_atom)[0]
            neighbor_idx_map = {sym: np.where(atomic_symbols == sym)[0] for sym in neighbor_atoms}

            for frame in coords:
                for c in central_idx:
                    rc = frame[c]
                    neighs = {}
                    for sym, idxs in neighbor_idx_map.items():
                        if len(idxs) == 0:
                            neighs[sym] = np.array([], dtype=int)
                            continue
                        dvecs = frame[idxs] - rc
                        dists = np.linalg.norm(dvecs, axis=1)
                        neighs[sym] = idxs[dists <= cutoff]

                    s_idxs = neighs.get("S", np.array([], dtype=int))
                    o_idxs = neighs.get("O", np.array([], dtype=int))

                    # S-P-S
                    if len(s_idxs) >= 2:
                        for i in range(len(s_idxs)):
                            for j in range(i+1, len(s_idxs)):
                                angle = compute_angle_deg(rc, frame[s_idxs[i]], frame[s_idxs[j]])
                                if angle is not None:
                                    angles_dict["S-P-S"].append(angle)

                    # O-P-O
                    if len(o_idxs) >= 2:
                        for i in range(len(o_idxs)):
                            for j in range(i+1, len(o_idxs)):
                                angle = compute_angle_deg(rc, frame[o_idxs[i]], frame[o_idxs[j]])
                                if angle is not None:
                                    angles_dict["O-P-O"].append(angle)

                    # S-P-O
                    if len(s_idxs) >= 1 and len(o_idxs) >= 1:
                        for a in s_idxs:
                            for b in o_idxs:
                                angle = compute_angle_deg(rc, frame[a], frame[b])
                                if angle is not None:
                                    angles_dict["S-P-O"].append(angle)

    return angles_dict

def find_structures(root):
    structures = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "type.raw" in filenames and "type_map.raw" in filenames:
            structures.append(dirpath)
    return structures

def plot_and_save_combined_histogram(angles_all_structures, outdir, output_filename="Angle_distributions.png"):
    """
    Genera un unico grafico con tutte le distribuzioni di angoli con linea continua per tutti.
    """
    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    colors = {"S-P-S": "steelblue", "S-P-O": "darkorange", "O-P-O": "green"}
    linestyle = "-"  # linea continua per tutti

    for atype, angles in angles_all_structures.items():
        if not angles:
            print(f"Nessun angolo trovato per {atype}")
            continue

        hist, bin_edges = np.histogram(angles, bins=bins, density=True)
        hist_smooth = gaussian_filter1d(hist, sigma=2)

        plt.plot(
            bin_edges[:-1],
            hist_smooth,
            label=f"{atype}",
            color=colors.get(atype, "black"),
            linestyle=linestyle,
            linewidth=2
        )

    plt.xlabel("Angolo (°)", fontsize=14)
    plt.ylabel("Densità di probabilità", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.legend(loc="upper right", fontsize=12)

    output_path_file = os.path.join(outdir, output_filename)
    plt.tight_layout()
    plt.savefig(output_path_file, dpi=300)
    plt.close()
    print(f"✅ Grafico combinato salvato in: {output_path_file}")

# --- SCRIPT PRINCIPALE ---
if __name__ == "__main__":
    os.makedirs(output_path, exist_ok=True)
    structures = find_structures(dataset_path)
    print(f"Trovate {len(structures)} strutture.")

    angles_total = {atype: [] for atype in angle_types_to_make}
    for struct in structures:
        angs = process_structure_for_angles(struct)
        for atype in angle_types_to_make:
            angles_total[atype].extend(angs.get(atype, []))

    plot_and_save_combined_histogram(angles_total, output_path)
    print("Elaborazione angoli completata.")

