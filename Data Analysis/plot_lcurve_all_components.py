#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_all_components.py
Plots all DeepMD training metrics (Learning Rate, Global RMSE, Energy, Forces, 
and Virial RMSE) stacked vertically in 5 subplots sharing the same X-axis.
Author: Lorenzo-Atanasio-2000-hub
"""

import argparse
import pandas as pd
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

def apply_minimal_style(ax, title, ylabel):
    """Applies a uniform minimalistic style to a subplot."""
    ax.set_title(title, fontsize=12, pad=6)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # hide top and right frames
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
        
    ax.legend(loc='upper right', frameon=False, fontsize='small')

def plot_pair(ax, step, y_trn, y_val, title, ylabel, smooth_w=1, logscale=False):
    """Creates a subplot tracking Train/Validation RMSE components."""
    ytr = smooth(y_trn, smooth_w)
    yvl = smooth(y_val, smooth_w)
    
    ax.plot(step, ytr, linewidth=1.2, label="Train")
    ax.plot(step, yvl, linewidth=1.2, label="Validation")
    
    if logscale:
        ax.set_yscale('log')
        
    apply_minimal_style(ax, title, ylabel)

def plot_lr(ax, step, lr):
    """Plots the learning rate decay curve."""
    ax.plot(step, lr, linewidth=1.2, linestyle='-', color='gray', label="Learning rate")
    
    apply_minimal_style(ax, "Learning Rate Decay", "Learning Rate")

def main():
    p = argparse.ArgumentParser(description="Plot learning rate, global errors, E, F, V vs step")
    p.add_argument("file", help="Data log file (or - for stdin)")
    p.add_argument("-o", "--out", default="all_components.png", help="Output PNG image filename")
    p.add_argument("--smooth", type=int, default=1, help="Rolling average window size (default: 1)")
    p.add_argument("--log", action="store_true", help="Apply log scale to Y-axes (excludes Learning Rate)")
    p.add_argument("--show", action="store_true", help="Display plot interactively on screen")
    args = p.parse_args()

    df = read_data(args.file)
    step = df["step"]

    # 5 stacked subplots sharing X-axis
    fig, axs = plt.subplots(5, 1, figsize=(10, 16), sharex=True)

    # 1) Learning rate
    plot_lr(axs[0], step, df["lr"])

    # 2) Global Error
    plot_pair(axs[1], step, df["rmse_trn"], df["rmse_val"], "Global Error", "RMSE", args.smooth, args.log)

    # 3) Energies
    plot_pair(axs[2], step, df["rmse_e_trn"], df["rmse_e_val"], "Energy Component (RMSE_e)", "RMSE Energy", args.smooth, args.log)

    # 4) Forces
    plot_pair(axs[3], step, df["rmse_f_trn"], df["rmse_f_val"], "Forces Component (RMSE_f)", "RMSE Forces", args.smooth, args.log)

    # 5) Virial
    plot_pair(axs[4], step, df["rmse_v_trn"], df["rmse_v_val"], "Virial Component (RMSE_v)", "RMSE Virial", args.smooth, args.log)

    
    axs[-1].set_xlabel("Training Step", fontsize=12)
    
    fig.suptitle("Training Metrics Evolution", fontsize=15, fontweight='bold', y=0.98)
    
   
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(args.out, dpi=300)
    print(f" Unified metrics plot saved to: {args.out}")
    
    if args.show:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    main()
