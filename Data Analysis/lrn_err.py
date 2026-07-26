#!/usr/bin/env python3
"""
plot_learning_global.py
Plotta:
  - Learning curve (rmse_trn e rmse_val)
  - Errore globale del modello vs step
Con legenda unica in alto a destra e linee sottili.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def read_data(path):
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
    if window <= 1:
        return series
    return series.rolling(window=window, center=True, min_periods=1).mean()

def main():
    p = argparse.ArgumentParser(description="Plot learning curve + errore globale vs step")
    p.add_argument("file", help="file di dati (o - per stdin)")
    p.add_argument("-o","--out", default="learning_global.png", help="file immagine di output")
    p.add_argument("--smooth", type=int, default=1, help="rolling average window")
    p.add_argument("--log", action="store_true", help="scala y log")
    p.add_argument("--show", action="store_true", help="mostra grafico a video")
    args = p.parse_args()

    df = read_data(args.file)
    step = df["step"]
    rmse_trn = smooth(df["rmse_trn"], args.smooth)
    rmse_val = smooth(df["rmse_val"], args.smooth)
    lr = df["lr"]

    fig, ax = plt.subplots(figsize=(9,5))
    lines = []
    labels = []

    l1, = ax.plot(step, rmse_trn, linewidth=1, label='RMSE train')
    l2, = ax.plot(step, rmse_val, linewidth=1, label='RMSE val')
    lines += [l1, l2]
    labels += [l.get_label() for l in [l1, l2]]

    ax.set_xlabel("Step di training")
    ax.set_ylabel("RMSE")
    ax.grid(True, linestyle='--', alpha=0.4)
    if args.log:
        ax.set_yscale('log')

    ax2 = ax.twinx()
    l3, = ax2.plot(step, lr, linestyle='--', linewidth=2, color='gray', label='Learning rate')
    ax2.set_ylabel("Learning rate")
    lines.append(l3)
    labels.append(l3.get_label())

    # Legenda unica
    ax.legend(lines, labels, loc='upper right', frameon=False)

    fig.suptitle("Learning curve e errore globale vs step", fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(args.out, dpi=250)
    if args.show:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    main()

