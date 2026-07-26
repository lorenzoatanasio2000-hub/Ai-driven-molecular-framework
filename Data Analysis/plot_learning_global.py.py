#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_learning_global.py
Plots the training/validation learning curves (RMSE) on the left Y-axis 
and the learning rate decay on the right Y-axis using a dual-axes (twinx) layout.
Author: Lorenzo-Atanasio-2000-hub
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def read_data(path):
    """Reads DeepMD training log data file safely."""
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

def main():
    p = argparse.ArgumentParser(description="Plot learning curves (RMSE) and learning rate decay vs step.")
    p.add_argument("file", help="Data log file (or - for stdin)")
    p.add_argument("-o", "--out", default="learning_global.png", help="Output PNG image filename")
    p.add_argument("--smooth", type=int, default=1, help="Rolling average window size")
    p.add_argument("--log", action="store_true", help="Apply log scale to the left Y-axis (RMSE)")
    p.add_argument("--show", action="store_true", help="Display plot interactively on screen")
    args = p.parse_args()

    df = read_data(args.file)
    step = df["step"]
    rmse_trn = smooth(df["rmse_trn"], args.smooth)
    rmse_val = smooth(df["rmse_val"], args.smooth)
    lr = df["lr"]

    fig, ax = plt.subplots(figsize=(9, 5))
    lines = []
    labels = []

    # Primary Y-Axis (RMSE)
    l1, = ax.plot(step, rmse_trn, linewidth=1.2, label='RMSE Train')
    l2, = ax.plot(step, rmse_val, linewidth=1.2, label='RMSE Validation')
    lines += [l1, l2]
    labels += [l.get_label() for l in [l1, l2]]

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
  
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    if args.log:
        ax.set_yscale('log')

    # Secondary Y-Axis (Learning Rate)
    ax2 = ax.twinx()
    l3, = ax2.plot(step, lr, linestyle='--', linewidth=1.8, color='gray', label='Learning Rate')
    ax2.set_ylabel("Learning Rate", fontsize=12, color='dimgray')
    ax2.tick_params(axis='y', which='major', labelsize=11, labelcolor='dimgray')
    
    lines.append(l3)
    labels.append(l3.get_label())

    # Unified  Legend
    ax.legend(lines, labels, loc='upper right', frameon=False, fontsize=11)

    # hide top frames on both axes to clean up the layout
    for spine in ["top"]:
        ax.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)

    fig.suptitle("Learning Curves & Global Error Evolution", fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(args.out, dpi=250)
    print(f" Plot successfully saved to: {args.out}")
    
    if args.show:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    main()
