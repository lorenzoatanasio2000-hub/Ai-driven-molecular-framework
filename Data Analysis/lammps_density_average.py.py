#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lammps_density_average.py
Parses a LAMMPS log file, locates the 'Density' thermo column dynamically, 
and calculates the average density while optionally skipping equilibration steps.
Author: Lorenzo-Atanasio-2000-hub
"""

import argparse
import numpy as np
import sys

def calculate_lammps_density(filename, discard_percentage=10):
    densities = []
    density_index = None
    header_found = False

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue

                # Dynamically locate the 'Density' column header
                if "Density" in parts and not header_found:
                    density_index = parts.index("Density")
                    header_found = True
                    continue

                # Extract numerical data once the header is located
                if header_found:
                    try:
                        value = float(parts[density_index])
                        densities.append(value)
                    except (ValueError, IndexError):
                        continue
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    if not densities:
        print(" Warning: No numerical values found in the 'Density' column.")
        return

    # Statistical Analysis excluding equilibration phase 
    total_points = len(densities)
    skip_points = int(total_points * (discard_percentage / 100.0))
    production_data = densities[skip_points:]

    if not production_data:
        print("Error: Discard percentage is too high, no data left for analysis.")
        return

    mean_density = np.mean(production_data)
    std_density = np.std(production_data)

    print(f"Total steps found: {total_points}")
    print(f"Equilibration steps discarded ({discard_percentage}%): {skip_points}")
    print(f"Production steps analyzed: {len(production_data)}")
    print(f"Average Density: {mean_density:.6f} (± {std_density:.6f}) g/cm³ (or internal units)")

def main():
    parser = argparse.ArgumentParser(description="Calculate average density from LAMMPS log files.")
    parser.add_argument("file", help="LAMMPS log or thermo output file")
    parser.add_argument("--discard", type=int, default=10, 
                        help="Percentage of initial steps to discard as equilibration (default: 10)")
    args = parser.parse_args()

    calculate_lammps_density(args.file, args.discard)

if __name__ == "__main__":
    main()
