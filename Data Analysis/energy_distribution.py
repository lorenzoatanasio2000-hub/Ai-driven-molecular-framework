#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Energy Distribution KDE Analyzer for DeepMD datasets.
Parses all energy.npy files, normalizes total energies by the number of atoms 
(extracted from type.raw), and plots a unified Kernel Density Estimation (KDE).
Author: Lorenzo-Atanasio-2000-hub
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def count_atoms_in_structure(struct_path):
    """Counts the total number of atoms in a structure using type.raw."""
    try:
        type_file = os.path.join(struct_path, "type.raw")
        if os.path.exists(type_file):
            types = np.loadtxt(type_file, dtype=int)
            # If type.raw is a scalar (1 atom), return 1, otherwise return its length
            return types.size if types.ndim > 0 else 1
    except Exception as err:
        print(f"Warning: Could not read atoms count in {struct_path}: {err}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Plot unified energy distribution (KDE) normalized per atom.")
    parser.add_argument("--dataset", required=True, help="Root path of the dataset containing structures")
    parser.add_argument("--output", default=".", help="Directory where the PNG plot will be saved")
    args = parser.parse_args()

    all_normalized_energies = []

    # Recursively scan dataset and gather ALL energies normalized per atom
    for root, dirs, files in os.walk(args.dataset):
        if "energy.npy" in files and "type.raw" in files:
            n_atoms = count_atoms_in_structure(root)
            if n_atoms is None or n_atoms == 0:
                continue
                
            try:
                energy_file = os.path.join(root, "energy.npy")
                raw_energies = np.load(energy_file)
                
                # Normalize total energy to energy per atom (eV/atom)
                normalized_e = raw_energies / n_atoms
                all_normalized_energies.extend(normalized_e)
            except Exception as err:
                print(f"Error reading energies in {root}: {err}")

    if all_normalized_energies:
        all_normalized_energies = np.array(all_normalized_energies)

        fig, ax = plt.subplots(figsize=(10, 6))
        
        #  KDE Plot
        sns.kdeplot(
            all_normalized_energies,
            bw_adjust=0.3,    # Controls KDE smoothness
            fill=True,        # Colored area underneath
            color="steelblue",
            alpha=0.85,       # Solid fill 
            linewidth=1.2,    # Thin contour line
            ax=ax
        )

        # Axis formatting
        ax.set_xlabel("Energy (eV/atom)", fontsize=14)
        ax.tick_params(axis='x', which='major', labelsize=12)

        # Clean Y-Axis: completely hides values and tick marks 
        ax.set_ylabel("")
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        # remove top, right, and left frames
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)

        ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.6)

        # Save output plot safely using argparse paths
        output_file = os.path.join(args.output, "unified_energy_distribution_kde.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Energy KDE plot successfully saved to: {output_file}")

        plt.show()
    else:
        print("Error: No matching 'energy.npy' and 'type.raw' files found in the dataset.")

if __name__ == "__main__":
    main()
