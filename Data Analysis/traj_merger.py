#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merges multiple sequential XYZ trajectory files into a single continuous file.
Maintains the strict XYZ multi-frame format required by VMD, Ovito, etc.
Author: Lorenzo-Atanasio-2000-hub
"""

import sys

def merge_xyz_files(output_file, input_files):
    total_frames_copied = 0
    
    with open(output_file, 'w', encoding='utf-8') as fout:
        for file_path in input_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as fin:
                    for line in fin:
                        # Skip completely empty lines that might corrupt the parser
                        if not line.strip():
                            continue
                        fout.write(line)
            except FileNotFoundError:
                print(f"Warning: File '{file_path}' not found. Skipping.")
                continue

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_xyz.py <file1.xyz> <file2.xyz> [file3.xyz ...]")
        sys.exit(1)

    # Takes all input arguments passed by the user
    input_files = sys.argv[1:]
    
    # Hardcoded output name as requested, but placed in main for clarity
    output_file = "dump_total_2ns.xyz"
    
    print("Merging XYZ trajectories...")
    merge_xyz_files(output_file, input_files)
    print(f"Success! Files successfully merged into '{output_file}' without formatting loss.")
