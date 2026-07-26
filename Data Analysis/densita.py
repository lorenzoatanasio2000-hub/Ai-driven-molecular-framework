#!/usr/bin/env python3
import re
import sys

# --- Definizione masse atomiche (g/mol) ---
mass_Li = 6.941
mass_P  = 30.973762
mass_S  = 32.065
mass_O  = 15.999

# --- Funzione per leggere il volume dal file CP2K ---
def read_volume(cp2k_output):
    with open(cp2k_output, 'r') as f:
        for line in f:
            # Cerca la riga che contiene "CELL| Volume"
            if "CELL| Volume" in line:
                # Estrae l'ultimo numero della riga (il volume)
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if numbers:
                    return float(numbers[-1])
    raise ValueError("Volume non trovato nel file CP2K.")

# --- Funzione per contare gli atomi dallo XYZ ---
def count_atoms(xyz_file):
    counts = {'Li':0, 'P':0, 'S':0, 'O':0}
    with open(xyz_file, 'r') as f:
        lines = f.readlines()[2:]  # salto le prime due righe
        for line in lines:
            atom = line.split()[0]
            if atom in counts:
                counts[atom] += 1
    return counts

# --- Calcolo densità ---
def calculate_density(atom_counts, volume_ang3):
    total_mass_gmol = (atom_counts['Li']*mass_Li +
                       atom_counts['P']*mass_P +
                       atom_counts['S']*mass_S +
                       atom_counts['O']*mass_O)
    # Convertire in grammi
    total_mass_g = total_mass_gmol / 6.02214076e23
    # Volume in cm^3
    volume_cm3 = volume_ang3 * 1e-24
    density = total_mass_g / volume_cm3
    return float(f"{density:.6g}")

# --- Main ---
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Uso: {sys.argv[0]} <CP2K_output> <XYZ_file>")
        sys.exit(1)

    cp2k_file = sys.argv[1]
    xyz_file = sys.argv[2]

    volume = read_volume(cp2k_file)
    atoms = count_atoms(xyz_file)
    density = calculate_density(atoms, volume)

    print(f"Volume: {volume:.6g} Å^3")
    print(f"Atomi: {atoms}")
    print(f"Densità: {density:.6g} g/cm^3")

