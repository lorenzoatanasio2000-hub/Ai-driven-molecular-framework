#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration Space PCA Analyzer for DeepMD-style datasets.
Groups frames by chemical composition and performs PCA on flattened coordinates.
Note: Input coordinates should be rotationally and translationally aligned.
Author: Lorenzo-Atanasio-2000-hub
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import defaultdict, Counter

def load_types(struct_path):
    """Loads type.raw and type_map.raw from the structure directory."""
    try:
        types = np.loadtxt(os.path.join(struct_path, "type.raw"), dtype=int)
        with open(os.path.join(struct_path, "type_map.raw"), 'r', encoding='utf-8') as f:
            type_map = [line.strip() for line in f.readlines()]
        symbols = [type_map[t] for t in types]
        return types, symbols
    except Exception as e:
        print(f"Warning: Could not load types in {struct_path}: {e}")
        return None, None

def find_structures(root):
    """Recursively finds all directories containing training topology files."""
    structures = []
    for dirpath, _, filenames in os.walk(root):
        if "type.raw" in filenames and "type_map.raw" in filenames:
            structures.append(dirpath)
    return structures

def load_coordinates(struct_path, base_symbols):
    """Loads consistent frames from coord.npy and tracks expanded symbols for supercells."""
    frames = []
    final_symbols = base_symbols
    
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

            # Handle supercell expansions consistently
            if len(base_symbols) != n_atoms:
                if n_atoms % len(base_symbols) == 0:
                    repeat = n_atoms // len(base_symbols)
                    final_symbols = base_symbols * repeat  # Expand symbols list safely
                else:
                    continue

            for frame in coords:
                frames.append(frame.flatten())
        except Exception as e:
            print(f"Error parsing coordinates in {coord_path}: {e}")
            continue
            
    return frames, final_symbols

def composition_to_string(symbols):
    """Converts a list of symbols into a compact formula, e.g., ['Li','Li','P'] -> 'Li2P1'"""
    counter = Counter(symbols)
    return ''.join(f"{el}{counter[el]}" for el in sorted(counter))

def main():
    parser = argparse.ArgumentParser(description="PCA Dimensionality Reduction on DeepMD Trajectories")
    parser.add_argument("--dataset", required=True, help="Root path of the dataset containing structures")
    parser.add_argument("--output", default="./PCA_plots", help="Directory where PNG plots will be saved")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    all_structures = find_structures(args.dataset)
    print(f"Found {len(all_structures)} candidate structure directories.")

    composition_dict = defaultdict(list)

    # Gather frames grouped by true chemical composition
    for struct in all_structures:
        types, symbols = load_types(struct)
        if types is None:
            continue

        frames, expanded_symbols = load_coordinates(struct, symbols)
        if frames:
            comp_str = composition_to_string(expanded_symbols)
            composition_dict[comp_str].extend(frames)

    # Perform PCA and plot for each unique composition
    for comp_str, frames in composition_dict.items():
        X = np.array(frames)
        if len(X) < 2:
            print(f"Skipping {comp_str}: Not enough frames ({len(X)}) for PCA analysis.")
            continue

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], s=20, alpha=0.6, c="steelblue", edgecolors="none")
        
        ax.set_title(f"Configuration Space PCA - {comp_str}", fontsize=14, pad=15)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=12)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Clean look: remove top and right borders
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        output_file = os.path.join(args.output, f"PCA_{comp_str}.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        print(f"Plot successfully saved: {output_file}")

if __name__ == "__main__":
    main()
