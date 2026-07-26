#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combines multiple RDF (Radial Distribution Function) files (r, g(r)) from TraVis into one plot.
Automatically converts distances from picometers (pm) to Ångström (Å).
Author: Lorenzo-Atanasio-2000-hub
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_rdf(file):
    """Loads RDF data from a semicolon-separated file and converts pm to Å."""
    try:
        df = pd.read_csv(file, sep=';', comment='#', header=None)
        if df.shape[1] < 2:
            raise ValueError(f"{file} does not contain at least two columns.")
        
        r = df.iloc[:, 0].values / 100.0  # Convert pm -> Å
        g = df.iloc[:, 1].values
        return r, g
    except Exception as e:
        print(f"Error loading {file}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Combine multiple RDF files into a single plot.")
    parser.add_argument("files", nargs="+", help="RDF data files (r, g(r))")
    parser.add_argument("--output", default="combined_rdf.png", help="Output PNG filename")
    parser.add_argument("--colors", nargs="+", help="Custom colors for each curve (optional)")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively")
    parser.add_argument("--smooth", type=int, default=1, help="Rolling average window size for smoothing (optional)")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, file in enumerate(args.files):
        r, g = load_rdf(file)
        if r is None or g is None:
            continue

        # Apply smoothing if requested
        if args.smooth > 1:
            g = pd.Series(g).rolling(window=args.smooth, center=True, min_periods=1).mean().values

        # Safe color mapping 
        color = None
        if args.colors:
            color = args.colors[i % len(args.colors)]

        # label extraction (removes extension safely, extracts text after last underscore)
        filename_no_ext = os.path.splitext(os.path.basename(file))[0]
        label = filename_no_ext.split("_")[-1] if "_" in filename_no_ext else filename_no_ext

        # Plot the curve
        ax.plot(r, g, label=label, color=color, linewidth=1.5)

    # Plot formatting 
    ax.set_xlabel("r (Å)", fontsize=14)
    ax.set_ylabel("g(r)", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # remove top and right frames
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
        
    ax.legend(fontsize=12, loc="upper right", frameon=False)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f" Plot successfully saved to: {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    main()
