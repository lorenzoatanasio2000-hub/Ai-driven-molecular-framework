#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bond Angle Distribution Function (BADF) Analyzer for DeepMD datasets.
Dynamically extracts and groups bond angles (e.g., S-P-S, S-P-O) around a central atom within a cutoff.
Note: Input coordinates should be unwrapped (does not explicitly apply PBC box wraps).
Author: Lorenzo-Atanasio-2000-hub
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from itertools import combinations_with_replacement, product
import sys

def load_types(structure_path):
    """Loads atom types and chemical element maps safely."""
    try:
        type_raw_path = os.path.join(structure_path, "type.raw")
        type_map_path = os.path.join(structure_path, "type_map.raw")
        types = np.loadtxt(type_raw_path, dtype=int)
        if types.ndim == 0:
            types = np.array([types])
        with open(type_map_path, "r", encoding="utf-8") as f:
            type_map = np.array([line.strip() for line in f.readlines()])
        return types, type_map
    except Exception as e:
        print(f"Warning: Could not load topology in {structure_path}: {e}")
        return None, None

def generate_angle_headers(central, neighbors):
    """Dynamically generates angle triplet string identifiers from neighbor list."""
    headers = []
    # Homogeneous pairs (e.g., S-P-S, O-P-O)
    for n in sorted(neighbors):
        headers.append(f"{n}-{central}-{n}")
    # Heterogeneous pairs (e.g., S-P-O)
    for n1, n2 in combinations_with_replacement(sorted(neighbors), 2):
        if n1 != n2:
            headers.append(f"{n1}-{central}-{n2}")
    return headers

def process_structure_vectorized(structure_path, central_atom, neighbor_atoms, cutoff):
    """Parses coordinates and vectorially extracts bond angles around central species."""
    types, type_map = load_types(structure_path)
    angle_types = generate_angle_headers(central_atom, neighbor_atoms)
    local_angles = {atype: [] for atype in angle_types}

    if types is None:
        return local_angles

    sets = [d for d in os.listdir(structure_path) if d.startswith("set.")]
    for s in sets:
        set_path = os.path.join(structure_path, s)
        npy_files = [f for f in os.listdir(set_path) if f.endswith(".npy") and 'coord' in f.lower()]

        for npy_file in npy_files:
            try:
                coords = np.load(os.path.join(set_path, npy_file))
                # coords from DeepMD raw/npy sets is (n_frames, n_atoms*3)
                if coords.ndim != 2 or coords.shape[1] % 3 != 0:
                    continue

                n_atoms_coord = coords.shape[1] // 3
                n_types = len(types)

                if n_atoms_coord != n_types:
                    if n_atoms_coord % n_types == 0:
                        types_expanded = np.tile(types, n_atoms_coord // n_types)
                    else:
                        continue
                else:
                    types_expanded = types

                coords = coords.reshape(coords.shape[0], n_atoms_coord, 3)
                atomic_symbols = type_map[types_expanded]

                c_idxs = np.where(atomic_symbols == central_atom)[0]
                if len(c_idxs) == 0:
                    continue

                # Pre-calculate neighbor indices arrays
                neigh_idxs = {sym: np.where(atomic_symbols == sym)[0] for sym in neighbor_atoms}

                # Vectorized processing per frame
                for frame in coords:
                    for c in c_idxs:
                        rc = frame[c]

                        # Gather neighbors within cutoff sphere
                        valid_neighbors = {}
                        for sym in neighbor_atoms:
                            idxs = neigh_idxs[sym]
                            if idxs.size == 0:
                                valid_neighbors[sym] = np.empty((0, 3))
                                continue
                            dvecs = frame[idxs] - rc
                            dists = np.linalg.norm(dvecs, axis=1)
                            valid_neighbors[sym] = frame[idxs[dists <= cutoff]]

                        # 1. Homogeneous T-C-T Angles (e.g., S-P-S)
                        for sym in neighbor_atoms:
                            arr = valid_neighbors[sym]
                            atype = f"{sym}-{central_atom}-{sym}"
                            if len(arr) >= 2:
                                # Multi-combination vectorization for triplets
                                for vec1, vec2 in combinations_with_replacement(arr, 2):
                                    if np.array_equal(vec1, vec2):
                                        continue
                                    v1, v2 = vec1 - rc, vec2 - rc
                                    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                                    local_angles[atype].append(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

                        # 2. Heterogeneous T1-C-T2 Angles (e.g., S-P-O)
                        for sym1, sym2 in combinations_with_replacement(sorted(neighbor_atoms), 2):
                            if sym1 == sym2:
                                continue
                            arr1, arr2 = valid_neighbors[sym1], valid_neighbors[sym2]
                            atype = f"{sym1}-{central_atom}-{sym2}"
                            if len(arr1) >= 1 and len(arr2) >= 1:
                                for vec1, vec2 in product(arr1, arr2):
                                    v1, v2 = vec1 - rc, vec2 - rc
                                    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                                    local_angles[atype].append(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

            except Exception as e:
                print(f"Error parsing file {npy_file}: {e}")
                continue

    return local_angles

def main():
    parser = argparse.ArgumentParser(description="Dynamically compute bond angle distributions for DeepMD files.")
    parser.add_argument("--dataset", required=True, help="Root folder of the configuration dataset")
    parser.add_argument("--output", default=".", help="Directory to save the combined output plot")
    parser.add_argument("--center", default="P", help="Central atom symbol (default: P)")
    parser.add_argument("--neighbors", nargs="+", default=["S", "O"], help="Neighboring species (default: S O)")
    parser.add_argument("--cutoff", type=float, default=2.6, help="Coordination sphere cutoff radius in Å (default: 2.6)")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian smoothing window factor (default: 2.0)")
    args = parser.parse_args()

    # Find structures recursively
    structures = []
    for dirpath, _, filenames in os.walk(args.dataset):
        if "type.raw" in filenames and "type_map.raw" in filenames:
            structures.append(dirpath)

    print(f"Located {len(structures)} active structure paths.")
    angle_types = generate_angle_headers(args.center, args.neighbors)
    angles_total = {atype: [] for atype in angle_types}

    for struct in structures:
        angs = process_structure_vectorized(struct, args.center, args.neighbors, args.cutoff)
        for atype in angle_types:
            angles_total[atype].extend(angs.get(atype, []))

    # --- Plotting Architecture ---
    bins = np.linspace(0, 180, 181)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Adaptive distinct color palette map
    cmap = plt.cm.get_cmap("tab10")

    plot_executed = False
    for i, (atype, angles) in enumerate(angles_total.items()):
        if not angles:
            print(f"No valid triplets registered for angle type: {atype}")
            continue

        hist, bin_edges = np.histogram(angles, bins=bins, density=True)
        hist_smooth = gaussian_filter1d(hist, sigma=args.sigma)

        ax.plot(bin_edges[:-1], hist_smooth, linewidth=2, label=atype, color=cmap(i % 10))
        plot_executed = True

    if not plot_executed:
        print(" Error: No bond angles could be extracted under current parameters.")
        sys.exit(1)

    ax.set_xlabel("Angle (°)", fontsize=14)
    ax.set_ylabel("Probability Density", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim(0, 180)

    # Clean framing setup
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc="upper right", frameon=False, fontsize=12)

    output_file = os.path.join(args.output, f"angle_distributions_{args.center}.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f" Combined angle distribution plot successfully saved to: {output_file}")

if __name__ == "__main__":
    main()
