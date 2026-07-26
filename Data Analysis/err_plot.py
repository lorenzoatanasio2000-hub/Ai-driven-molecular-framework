#!/usr/bin/env python3
"""
plot_components.py
Plotta 3 subplot:
  - Energie (rmse_e_trn / rmse_e_val)
  - Forze (rmse_f_trn / rmse_f_val)
  - Viriale (rmse_v_trn / rmse_v_val)
Ogni subplot ha la sua legenda in alto a destra.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def read_data(path):
    """Legge il file DeepMD di training"""
    if path == "-":
        df = pd.read_csv(sys.stdin, comment='#', delim_whitespace=True, header=None)
    else:
        df = pd.read_csv(path, comment='#', delim_whitespace=True, header=None)
    cols = ["step","rmse_val","rmse_trn","rmse_e_val","rmse_e_trn",
            "rmse_f_val","rmse_f_trn","rmse_v_val","rmse_v_trn","lr"]
    df = df.iloc[:, :len(cols)]
    df.columns = cols
    return df

def smooth(series, window):
    """Applica media mobile"""
    if window <= 1:
        return series
    return series.rolling(window=window, center=True, min_periods=1).mean()

def plot_pair(ax, step, y_trn, y_val, title, ylabel, smooth_window=1, logscale=False):
    """Crea un singolo subplot con legenda locale"""
    ytrn = smooth(y_trn, smooth_window)
    yval = smooth(y_val, smooth_window)
    ax.plot(step, ytrn, linewidth=1, label="Train")
    ax.plot(step, yval, linewidth=1, label="Validation")
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.35)
    if logscale:
        ax.set_yscale('log')
    ax.legend(loc='upper right', frameon=False, fontsize='small')

def main():
    p = argparse.ArgumentParser(description="Plot componenti di errore DeepMD (E, F, V) vs step")
    p.add_argument("file", help="file di dati (o - per stdin)")
    p.add_argument("-o","--out", default="components.png", help="file immagine di output")
    p.add_argument("--smooth", type=int, default=1, help="rolling average window")
    p.add_argument("--log", action="store_true", help="scala log per asse y")
    p.add_argument("--show", action="store_true", help="mostra grafico a video")
    args = p.parse_args()

    df = read_data(args.file)
    step = df["step"]

    # 3 subplot (energie, forze, viriale)
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    plot_pair(axs[0], step, df["rmse_e_trn"], df["rmse_e_val"],
              "Energie", "RMSE energia", args.smooth, args.log)
    plot_pair(axs[1], step, df["rmse_f_trn"], df["rmse_f_val"],
              "Forze", "RMSE forze", args.smooth, args.log)
    plot_pair(axs[2], step, df["rmse_v_trn"], df["rmse_v_val"],
              "Viriale", "RMSE viriale", args.smooth, args.log)

    fig.suptitle("Errori per componente vs step di training", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.out, dpi=250)
    if args.show:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    main()

