#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combines different ADF (Angular Distribution Function) files into a single plot.
Limits the X-axis from 75° to 150° and hides the Y-axis for comparative analysis.
Author: Lorenzo-Atanasio-2000-hub
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_adf(file):
    try:
        # Handles semicolon-separated ADF output files safely
        df = pd.read_csv(file, sep=';', comment='#', header=None)
        if df.shape[1] < 2:
            raise ValueError(f"{file} does not contain at least two columns.")
        angle = df.iloc[:, 0].values
        occurrence = df.iloc[:, 1].values
        return angle, occurrence
    except Exception as e:
        print(f"Error loading {file}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Combine multiple ADF files into a single plot.")
    parser.add_argument("files", nargs="+", help="ADF files (angle, occurrence)")
    parser.add_argument("--output", default="combined_adf.png", help="Output PNG filename")
    parser.add_argument("--colors", nargs="+", help="Custom colors for each curve (optional)")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively")
    parser.add_argument("--smooth", type=int, default=1, help="Rolling average window size for smoothing (optional)")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, file in enumerate(args.files):
        angle, occurrence = load_adf(file)
        if angle is None or occurrence is None:
            continue

        # Apply rolling average smoothing if requested
        if args.smooth > 1:
            occurrence = pd.Series(occurrence).rolling(window=args.smooth, center=True, min_periods=1).mean().values

        # Robust color matching preventing IndexError
        color = None
        if args.colors:
            color = args.colors[i % len(args.colors)] # Wraps around if fewer colors than files

        # Smart label extraction: extracts name without extension
        filename_only = os.path.splitext(os.path.basename(file))[0]
        label = filename_only.split("_")[-1] if "_" in filename_only else filename_only

        # Plots the line and captures its properties
        line, = ax.plot(angle, occurrence, label=label, color=color, linewidth=1.5)
        
        # Uses the exact same color of the line for the filled area underneath
        ax.fill_between(angle, 0, occurrence, color=line.get_color(), alpha=0.2)

    # X-axis formatting
    ax.set_xlim(75, 150)
    ax.set_xlabel("Angle (°)", fontsize=14)
    ax.tick_params(axis='x', which='major', labelsize=12)

    # Clean Y-axis styling (hides ticks and labels for better comparative views)
    ax.set_ylabel("")
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    # Removes the top, right, and left frame boxes for a modern minimalistic look
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    # Grid and Legend
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=12, loc="upper right", frameon=False)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Plot successfully saved to: {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    main()
