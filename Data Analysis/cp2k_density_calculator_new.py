#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Density calculator from CP2K output and XYZ files.
Author: Lorenzo-Atanasio-2000-hub
"""

import re
import sys

# --- Atomic Masses (g/mol) ---
MASS_LI = 6.941
MASS_P  = 30.973762
MASS_S  = 32.065
MASS_O  = 15.999

# --- Read Last Volume from CP2K Output ---
def read_volume(cp2k_output):
    last_volume = None
    with open(cp2k_output, 'r', encoding='utf-8') as f:
        for line in f:
            if "CELL| Volume" in line:
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if numbers:
                    last_volume = float(numbers[-1])
                    
    if last_volume is None:
        raise ValueError("Error: 'CELL| Volume' not found in the CP2K file.")
    return last_volume

# --- Count Atoms from XYZ File ---
def count_atoms(xyz_file):
    counts = {'Li': 0, 'P': 0, 'S': 0, 'O': 0}
    with open(xyz_file, 'r', encoding='utf-8') as f:
        # Skip the first two header lines efficiently
        next(f)
        next(f)
        for line in f:
            split_line = line.split()
            if not split_line:
                continue
            
            # Cleans names like 'Li12' or 'P_1' extracting only the chemical symbol
            raw_atom = split_line[0]
            atom_match = re.match(r"([a-zA-Z]+)", raw_atom)
            
            if atom_match:
                atom = atom_match.group(1).capitalize() # Normalizes to 'Li', 'P', etc.
                if atom in counts:
                    counts[atom] += 1
    return counts

# --- Density Calculation ---
def calculate_density(atom_counts, volume_ang3):
    total_mass_gmol = (atom_counts['Li'] * MASS_LI +
                       atom_counts['P'] * MASS_P +
                       atom_counts['S'] * MASS_S +
                       atom_counts['O'] * MASS_O)
    
    # Convert to grams using Avogadro's number
    total_mass_g = total_mass_gmol / 6.02214076e23
    # Volume in cm³ (1 Å³ = 1e-24 cm³)
    volume_cm3 = volume_ang3 * 1e-24
    
    density = total_mass_g / volume_cm3
    return float(f"{density:.6g}")

# --- Main Execution ---
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <CP2K_output> <XYZ_file>")
        sys.exit(1)

    cp2k_file = sys.argv[1]
    xyz_file = sys.argv[2]

    try:
        volume = read_volume(cp2k_file)
        atoms = count_atoms(xyz_file)
        density = calculate_density(atoms, volume)

        print(f"Final Volume: {volume:.6g} Å³")
        print(f"Atom Counts: {atoms}")
        print(f"Calculated Density: {density:.6g} g/cm³")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(e)
        sys.exit(1)
