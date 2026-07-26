#!/usr/bin/env python3
"""
Combines different ADF files (angle, occurrence) in one PNG file. 
X axis is limited to 75° - 150°.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_adf(file):
    try:
        df = pd.read_csv(file, sep=';', comment='#', header=None)
        if df.shape[1] < 2:
            raise ValueError(f"{file} doesn't have at least two columns")
        angle = df.iloc[:,0].values
        occurrence = df.iloc[:,1].values
        return angle, occurrence
    except Exception as e:
        print(f"Error loading {file}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Combina più ADF in un unico grafico")
    parser.add_argument("files", nargs="+", help="File ADF (angle, occurrence)")
    parser.add_argument("--output", default="combined_adf.png", help="Nome file PNG di output")
    parser.add_argument("--colors", nargs="+", help="Colori per ogni curva (opzionale)")
    parser.add_argument("--show", action="store_true", help="Mostra il grafico a video")
    parser.add_argument("--smooth", type=int, default=1, help="Media mobile (opzionale)")
    args = parser.parse_args()

    plt.figure(figsize=(8,6))

    for i, file in enumerate(args.files):
        angle, occurrence = load_adf(file)
        if angle is None or occurrence is None:
            continue

        # smoothing if needed
        if args.smooth > 1:
            occurrence = pd.Series(occurrence).rolling(window=args.smooth, center=True, min_periods=1).mean().values

        # optional color
        color = None
        if args.colors and i < len(args.colors):
            color = args.colors[i]

        # file name
        label = os.path.basename(file).split("_")[-1].replace(".csv","")

        plt.plot(angle, occurrence, label=label, color=color)
        plt.fill_between(angle, 0, occurrence, color=color, alpha=0.3)

    # X axis limit
    plt.xlim(75, 150)

   
    plt.ylabel("")                     
    plt.tick_params(axis='y',          
                    which='both',
                    labelleft=False,
                    left=True,        
                    length=0)     

    
    plt.grid(True, linestyle='--', alpha=0.6)

    
    plt.xlabel("Angle (°)", fontsize=14)
    plt.tick_params(axis='x', which='major', labelsize=12)

    plt.legend(fontsize=12, loc="upper right")
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Plot saved in: {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    main()

