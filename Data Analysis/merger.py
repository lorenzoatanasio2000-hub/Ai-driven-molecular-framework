#!/usr/bin/env python3
import sys

def merge_xyz_files(output_file, input_files):
    with open(output_file, 'w') as fout:
        first_file = True
        for file in input_files:
            with open(file, 'r') as fin:
                for i, line in enumerate(fin):
                    # Salta righe completamente vuote
                    if line.strip() == "":
                        continue
                    # Salta la prima riga (numero di atomi) dei file successivi
                    if i == 0 and not first_file:
                        continue
                    fout.write(line)
            first_file = False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python merge_xyz.py file1.xyz file2.xyz ...")
        sys.exit(1)

    input_files = sys.argv[1:]  # l’ordine che dai viene rispettato
    output_file = "dump_total_2ns.xyz"
    merge_xyz_files(output_file, input_files)
    print(f"File uniti in '{output_file}' senza righe vuote e senza modifiche ai file originali")

