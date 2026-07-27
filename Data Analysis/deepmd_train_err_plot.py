#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plots DeepMD training error components (Energy, Forces, Virial RMSE) 
stacked vertically in 3 subplots sharing the same X-axis.
Author: Lorenzo-Atanasio-2000-hub
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sys

def read_data(path):
    """Reads DeepMD training data log file safely."""
    if path == "-":
        df = pd.read_csv(sys.stdin, comment='#', sep=r"\s+", header=None)
    else:
        df = pd.read_csv(path, comment='#', sep=r"\s+", header=None, encoding='utf-8')
        
    cols = ["step","rmse_val","rmse_trn","rmse_e_val","rmse_e_trn",
            "rmse_f_val","rmse_f_trn","rmse_v_val","rmse_v_trn","lr"]
    df = df.iloc[:, :len(cols)]
    df.columns = cols
    return df

def smooth(series, window):
    """Applies a rolling average window for data smoothing."""
    if window <= 1:
        return series
    return series.rolling(window=window, center=True, min_periods=1).mean()

def plot_pair(ax, step, y_trn, y_val, title, ylabel, smooth_window=1, logscale=False):
    """Creates a single subplot with its local legend and grid."""
    ytrn = smooth(y_trn, smooth_window)
    yval = smooth(y_val, smooth_window)
    
    ax.plot(step, ytrn, linewidth=1.2, label="Train")
    ax.plot(step, yval, linewidth=1.2, label="Validation")
    
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    if logscale:
        ax.set_yscale('log')
        
    # Clean legend
    ax.legend(loc='upper right', frameon=False, fontsize='small')
    
    # hide top and right frames
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

def main():
    p = argparse.ArgumentParser(description="Plot DeepMD error components (E, F, V) vs training step")
    p.add_argument("file", help="Data file (or - for stdin)")
    p.add_argument("-o", "--out", default="components.png", help="Output image filename")
    p.add_argument("--smooth", type=int, default=1, help="Rolling average window size")
    p.add_argument("--log", action="store_true", help="Apply log scale to Y-axis")
    p.add_argument("--show", action="store_true", help="Display plot interactively on screen")
    args = p.parse_args()

    df = read_data(args.file)
    step = df["step"]

    # 3 stacked subplots sharing X-axis
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    plot_pair(axs[0], step, df["rmse_e_trn"], df["rmse_e_val"], "Energy Component", "RMSE Energy", args.smooth, args.log)
    plot_pair(axs[1], step, df["rmse_f_trn"], df["rmse_f_val"], "Forces Component", "RMSE Forces", args.smooth, args.log)
    plot_pair(axs[2], step, df["rmse_v_trn"], df["rmse_v_val"], "Virial Component", "RMSE Virial", args.smooth, args.log)

    # Set X-label ONLY on the bottom-most subplot 
    axs[2].set_xlabel("Training Step", fontsize=12)

    fig.suptitle("Error Components vs Training Step", fontsize=14, fontweight='bold')
    
    # Adjusts spacing keeping room for the main title
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.savefig(args.out, dpi=250)
    print(f"Plot successfully saved to: {args.out}")
    
    if args.show:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    main()
