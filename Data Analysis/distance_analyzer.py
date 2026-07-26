#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimum Interatomic Distance Distribution Analyzer for DeepMD datasets.
Extracts the shortest bond distances between a reference atom (e.g., Li) and a target list (e.g., O).
Note: Assumes unwrapped coordinates or large cells (does not explicitly apply PBC box wraps).
Author: Lorenzo-Atanasio-2000-hub
Usage: python distance_analyzer.py --dataset ./dataset --ref Li --targets O S P
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import sys

def load_types(structure_path):
    """Loads atom types and the corresponding chemical element map."""
    try:
        type_raw_path = os.path.join(structure_path, "type.raw")
        type_map_path = os.path.join(structure_path, "type_map.raw")

        types = np.loadtxt(type_raw_path, dtype=int)
        # Ensure types is always treated as a 1D array even if it contains 1 atom
        if types.ndim == 0:
            types = np.array([types])

        with open(type_map_path, "r", encoding="utf-8") as f:
            type_map = [line.strip() for line in f.readlines()]

        return types, np.array(type_map)
    except Exception as e:
        print(f"Error loading topology in {structure_path}: {e}")
        return None, None

def get_min_distances_vectorized(coords, types_expanded, type_map, atom1, atom2):
    """Calculates minimum distances using fast NumPy broadcasting over frames."""
    atomic_symbols = type_map[types_expanded]
    idx1 = np.where(atomic_symbols == atom1)[0]
    idx2 = np.where(atomic_symbols == atom2)[0]

    if len(idx1) == 0 or len(idx2) == 0:
        return []

    # Extract coordinates for all targeted sub-atoms across all frames
    # Shape: (n_frames, n_atoms_1, 3) and (n_frames, n_atoms_2, 3)
    c1 = coords[:, idx1, :]
    c2 = coords[:, idx2, :]

    # Vectorized distance matrix calculation via broadcasting
    # Shape of diff: (n_frames, n_atoms_1, n_atoms_2, 3)
    diff = c1[:, :, np.newaxis, :] - c2[:, np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)

    # Find the minimum distance to any atom2 for each atom1, then flatten across all frames
    min_dists = np.min(dists, axis=-1).flatten()
    return min_dists.tolist()

def process_structure(structure_path, atom1, atom2):
    """Processes coordinate sets within a structure directory vectorially."""
    types, type_map = load_types(structure_path)
    all_dists = []
    
    if types is None:
        return all_dists

    sets = [d for d in os.listdir(structure_path) if d.startswith("set.")]
    for s in sets:
        set_path = os.path.join(structure_path, s)
        npy_files = [f for f in os.listdir(set_path) if f.endswith(".npy") and 'coord' in f.lower()]

        for npy_file in npy_files:
            try:
                coords = np.load(os.path.join(set_path, npy_file))
                if coords.ndim != 2 or coords.shape[1] % 3 != 0:
                    continue

                n_atoms_coord = coords.shape[1] // 3
                n_types = len(types)

                if n_atoms_coord != n_types:
                    if n_atoms_coord % n_types == 0:
                        repeat_factor = n_atoms_coord // n_types
                        types_expanded = np.tile(types, repeat_factor)
                    else:
                        continue
                else:
                    types_expanded = types

                # Reshape to (n_frames, n_atoms, 3)
                coords = coords.reshape(coords.shape[0], n_atoms_coord, 3)
                
                # Compute vectorially for the entire set file at once
                dists = get_min_distances_vectorized(coords, types_expanded, type_map, atom1, atom2)
                all_dists.extend(dists)
                
            except Exception as e:
                print(f"Error reading coordinates {npy_file}: {e}")
                continue

    return all_dists

def main():
    parser = argparse.ArgumentParser(description="Analyze and plot minimum interatomic distance distributions.")
    parser.add_argument("--dataset", required=True, help="Root path of the dataset containing structures")
    parser.add_argument("--output", default=".", help="Directory where the PNG plot will be saved")
    parser.add_argument("--ref", default="Li", help="Reference atom symbol (default: Li)")
    parser.add_argument("--targets", nargs="+", default=["O"], help="List of target atom symbols (default: O)")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian smoothing factor (default: 2.0)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    # Find all structure folders containing topology raw files
    structures = []
    for dirpath, _, filenames in os.walk(args.dataset):
        if "type.raw" in filenames and "type_map.raw" in filenames:
            structures.append(dirpath)

    print(f"Found {len(structures)} structure directories.")
    all_distances = {atom2: [] for atom2 in args.targets}

    for struct in structures:
        for atom2 in args.targets:
            dists = process_structure(struct, args.ref, atom2)
            all_distances[atom2].extend(dists)

    if all(len(all_distances[atom2]) == 0 for atom2 in args.targets):
        print("❌ Error: No valid minimum bonds found across the entire dataset.")
        sys.exit(1)

    # --- Plotting Configuration ---
    bins = np.linspace(0, 5, 100)
    fig, ax = plt.subplots(figsize=(8, 5))

    for atom2 in args.targets:
        data = all_distances[atom2]
        if len(data) == 0:
            print(f"Warning: No minimum connections found for pair {args.ref}-{atom2}")
            continue
            
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        hist_smooth = gaussian_filter1d(hist, sigma=args.sigma)
        ax.plot(bin_edges[:-1], hist_smooth, linewidth=1.5, label=f'{args.ref}–{atom2} closest dist')

    ax.set_xlabel("Minimum Distance (Å)", fontsize=13)
    ax.set_ylabel("Probability Density", fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # remove top and right frame lines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
        
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=11, loc="upper right", frameon=False)

    #  output naming combining targeted species dynamically
    targets_str = "_".join(args.targets)
    filename = f"min_distances_{args.ref}_{targets_str}.png"
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, filename), dpi=300)
    plt.close(fig)
    print(f" Distribution plot successfully saved to: {os.path.join(args.output, filename)}")

if __name__ == "__main__":
    main()
