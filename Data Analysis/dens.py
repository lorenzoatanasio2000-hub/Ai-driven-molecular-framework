#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Script per calcolare la densità da un file di output
# Modificato per usare SOLO l'ultimo volume trovato

import sys

if len(sys.argv) < 2:
    print("Uso: python densita.py file.out")
    sys.exit(1)

outfile = sys.argv[1]

# Molle molari in g/mol
masse = {
    "Li": 6.941,
    "P": 30.973761998,
    "S": 32.065,
    "O": 15.9994,
}

# Conta degli atomi (adatta questi valori alla tua struttura)
n_atoms = {
    "Li": 12,
    "P": 4,
    "S": 8,
    "O": 4,
}

# Calcolo massa totale in g/mol
massa_tot = sum(masse[el] * n_atoms[el] for el in n_atoms)

# Conversione: 1 mol = 6.022e23 particelle
Na = 6.02214076e23

# Conversione: 1 Å³ = 1e-24 cm³
fattore_vol = 1.0e-24

# ---- Lettura del file .out e ricerca ultimo volume ----
ultimo_volume = None

with open(outfile) as f:
    for line in f:
        if "CELL| Volume" in line:
            try:
                ultimo_volume = float(line.split()[-1].replace(",", ""))
            except ValueError:
                continue

if ultimo_volume is None:
    print("Errore: nessun volume trovato nel file!")
    sys.exit(1)

print(f"Ultimo volume trovato (Å^3): {ultimo_volume:.6f}")

# ---- Calcolo densità ----
# massa_tot (g/mol) / Na -> massa della cella (g)
# volume (Å³) -> volume in cm³
massa_cella = massa_tot / Na  # grammi
volume_cm3 = ultimo_volume * fattore_vol

densita = massa_cella / volume_cm3  # g/cm³

print(f"Densità calcolata: {densita:.6f} g/cm^3")

