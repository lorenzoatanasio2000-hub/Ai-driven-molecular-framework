#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Force Norms Distribution Analyzer for DeepMD datasets.
Parses force.npy files across simulation types, reshapes split Cartesian components,
calculates atomic force magnitudes, and overlays safe KDE/Histogram distributions.
Author: Lorenzo-Atanasio-2000-hub
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import sys

def main():
    parser = argparse.ArgumentParser(description="Plot force magnitude distributions grouped by simulation type.")
    parser.add_argument("--dataset", required=True, help="Root path of the dataset containing structures")
    parser.add_argument("--output", default=".", help="Directory where the PNG plot will be saved")
    parser.add_argument("--xlim-max", type=float, default=15.0, help="Maximum limit for the X axis (default: 15.0)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Dictionary to store force magnitudes by simulation type
    force_norms_by_type = defaultdict(list)
    file_count = 0

    # Traverse directories and collect data from force.npy
    for root, _, files in os.walk(args.dataset):
        if "force.npy" in files:
            force_file = os.path.join(root, "force.npy")
            try:
                forces = np.load(force_file)
                file_count += 1
                
                if forces.ndim == 3:
                    norms = np.linalg.norm(forces, axis=2).flatten()
                elif forces.ndim == 2:
                    # Reshape to group Cartesian coordinates (X, Y, Z) per individual atom
                    reshaped_forces = forces.reshape(-1, 3)
                    norms = np.linalg.norm(reshaped_forces, axis=1)
                else:
                    raise ValueError(f"Unrecognized force.npy shape: {forces.shape}")

                # Extract simulation type from the top-level subfolder name
                rel_path = os.path.relpath(root, args.dataset)
                sim_type = rel_path.split(os.sep)[0] if rel_path != "." else "root"

                force_norms_by_type[sim_type].extend(norms.tolist())

            except Exception as err:
                print(f"Error parsing forces in {root}: {err}")

    if not force_norms_by_type:
        print("❌ Error: No 'force.npy' files found in the specified dataset path.")
        sys.exit(1)

    # Clean and filter numerical data
    for sim_type in sorted(force_norms_by_type.keys()):
        arr = np.array(force_norms_by_type[sim_type], dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        arr = arr[np.isfinite(arr)]
        force_norms_by_type[sim_type] = arr

    # --- Plotting Configuration ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    types = sorted(force_norms_by_type.keys())
    palette = sns.color_palette(n_colors=max(1, len(types)))

    for i, sim_type in enumerate(types):
        data = force_norms_by_type[sim_type]
        if data.size == 0:
            continue

        unique_vals = np.unique(data)
        
        # Adaptive plotting: use histogram if data is scarce or uniform to avoid KDE crashes
        if unique_vals.size < 2 or data.size < 5:
            sns.histplot(
                data,
                bins=min(10, max(1, data.size)),
                stat='density',
                alpha=0.5,
                label=sim_type,
                color=palette[i],
                ax=ax
            )
            if unique_vals.size == 1:
                ax.axvline(unique_vals[0], color=palette[i], linestyle='--', linewidth=1)
        else:
            sns.kdeplot(
                data,
                bw_adjust=0.3,
                fill=True,
                color=palette[i],
                alpha=0.5,
                label=sim_type,
                ax=ax
            )

    # Style Formatting
    ax.set_xlabel("Force Magnitude |F| (eV/Å)", fontsize=14)
    ax.set_ylabel("")
    ax.set_xlim(0, args.xlim_max)
    ax.tick_params(axis='x', which='major', labelsize=12)
    
    # completely hides tick numbers but keeps the axis structure
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    # hide top, right, and left frame strokes
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    # Unified grid lines
    ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=12)

    # Save and handle output cleanly
    output_file = os.path.join(args.output, "forces_distribution_kde.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f" Distribution plot successfully saved to: {output_file}")

    plt.show()

if __name__ == "__main__":
    main()
