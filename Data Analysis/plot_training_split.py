#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_training_split.py
Generates learning curves and error analysis plots from training log files.
Author: Lorenzo-Atanasio-2000-hub
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

def read_data(path):
    # Fixed deprecated delim_whitespace using sep=r"\s+"
    if path == "-":
        df = pd.read_csv(sys.stdin, comment="#", sep=r"\s+", header=None)
    else:
        df = pd.read_csv(path, comment="#", sep=r"\s+", header=None)
        
    cols = ["step","rmse_val","rmse_trn","rmse_e_val","rmse_e_trn",
            "rmse_f_val","rmse_f_trn","rmse_v_val","rmse_v_trn","lr"]
            
    if df.shape[1] < len(cols):
        raise ValueError(f"File contains {df.shape[1]} columns, but expected at least {len(cols)}.")
    
    df = df.iloc[:, :len(cols)]
    df.columns = cols
    return df

def smooth(series, window):
    if window <= 1:
        return series
    return series.rolling(window=window, center=True, min_periods=1).mean()

def plot_pair(ax, step, y_trn, y_val, title, ylabel, smooth_w=1, logscale=False):
    ytr = smooth(y_trn, smooth_w)
    yvl = smooth(y_val, smooth_w)
    ax.plot(step, ytr, linewidth=1.2, label="Train")
    ax.plot(step, yvl, linewidth=1.2, linestyle="-", label="Validation")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="-", alpha=0.35)
    if logscale:
        ax.set_yscale("log")
    ax.legend(loc="upper right", frameon=False, fontsize="small")

def plot_lr(ax, step, lr):
    ax.plot(step, lr, linewidth=1.2, linestyle="-", color="gray", label="Learning rate")
    ax.set_title("Learning rate")
    ax.set_ylabel("Learning rate")
    ax.grid(True, linestyle="-", alpha=0.35)
    ax.legend(loc="upper right", frameon=False, fontsize="small")

def format_axes(axes):
    """Uniformly formats fonts and labels across subplots."""
    for ax in axes:
        ax.set_xlabel("Training step", fontsize=14)
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.title.set_fontsize(16)
        leg = ax.get_legend()
        if leg:
            for text in leg.get_texts():
                text.set_fontsize(12)

def main():
    p = argparse.ArgumentParser(description="Split plot: LR+global, E/F/V")
    p.add_argument("file", help="data file (or - for stdin)")
    p.add_argument("--smooth", type=int, default=1, help="rolling average window")
    p.add_argument("--log", action="store_true", help="y-axis log scale")
    p.add_argument("--show", action="store_true", help="display plots on screen")
    args = p.parse_args()

    df = read_data(args.file)
    step = df["step"]

    # --- Figure 1: Learning rate + Global Error ---
    fig1, axs1 = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    plot_lr(axs1[0], step, df["lr"])
    plot_pair(axs1[1], step, df["rmse_trn"], df["rmse_val"], "Global Error", "RMSE", args.smooth, args.log)
    
    format_axes(axs1)
    fig1.tight_layout()
    fig1.savefig("learning_global.png", dpi=300)

    # --- Figure 2: Energy, Forces, Virial ---
    fig2 = plt.figure(figsize=(14, 9))
    gs = fig2.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.35, wspace=0.3)
    
    ax0 = fig2.add_subplot(gs[0, 0])  # Energy
    ax1 = fig2.add_subplot(gs[0, 1])  # Forces
    ax2 = fig2.add_subplot(gs[1, :])  # Virial (Spans 2 columns to center it)

    plot_pair(ax0, step, df["rmse_e_trn"], df["rmse_e_val"], "Energy (RMSE_e)", "RMSE Energy", args.smooth, args.log)
    plot_pair(ax1, step, df["rmse_f_trn"], df["rmse_f_val"], "Forces (RMSE_f)", "RMSE Forces", args.smooth, args.log)
    plot_pair(ax2, step, df["rmse_v_trn"], df["rmse_v_val"], "Virial (RMSE_v)", "RMSE Virial", args.smooth, args.log)

    format_axes([ax0, ax1, ax2])
    fig2.tight_layout()
    fig2.savefig("errors_EFV.png", dpi=300)

    # --- Unified Screen Display Handling ---
    if args.show:
        plt.show()  # Shows both figures beautifully at the exact same time
    else:
        plt.close(fig1)
        plt.close(fig2)

if __name__ == "__main__":
    main()
