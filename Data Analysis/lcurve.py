#!/usr/bin/env python3
"""
plot_all_components.py

Crea una figura con 5 subplot (ognuno con legenda in alto a destra):
  1) Learning rate vs training step
  2) Errore globale (rmse_trn / rmse_val)
  3) Energie (rmse_e_trn / rmse_e_val)
  4) Forze (rmse_f_trn / rmse_f_val)
  5) Viriale (rmse_v_trn / rmse_v_val)
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sys

def read_data(path):
    """Legge file whitespace-separated con righe commentate da #"""
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
    """Applica media mobile (rolling mean)"""
    if window <= 1:
        return series
    return series.rolling(window=window, center=True, min_periods=1).mean()

def plot_pair(ax, step, y_trn, y_val, title, ylabel, smooth_w=1, logscale=False):
    """Crea subplot con RMSE train/val + legenda locale"""
    ytr = smooth(y_trn, smooth_w)
    yvl = smooth(y_val, smooth_w)
    ax.plot(step, ytr, linewidth=1, label="Train")
    ax.plot(step, yvl, linewidth=1, label="Validation")
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='-', alpha=0.35)
    if logscale:
        ax.set_yscale('log')
    ax.legend(loc='upper right', frameon=False, fontsize='small')

def plot_lr(ax, step, lr):
    """Plotta solo il learning rate"""
    ax.plot(step, lr, linewidth=1, linestyle='-', color='gray', label="Learning rate")
    ax.set_title("Learning rate vs step di training")
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning rate")
    ax.grid(True, linestyle='-', alpha=0.35)
    ax.legend(loc='upper right', frameon=False, fontsize='small')

def main():
    p = argparse.ArgumentParser(description="Plot learning rate, errori globali, E, F, V vs step")
    p.add_argument("file", help="file di dati (o - per stdin)")
    p.add_argument("-o","--out", default="all_components.png", help="file immagine di output")
    p.add_argument("--smooth", type=int, default=1, help="rolling average window (default 1)")
    p.add_argument("--log", action="store_true", help="scala log per gli assi y (eccetto LR)")
    p.add_argument("--show", action="store_true", help="mostra il grafico a video")
    args = p.parse_args()

    df = read_data(args.file)
    step = df["step"]

    fig, axs = plt.subplots(5, 1, figsize=(10, 16), sharex=True)
    plt.subplots_adjust(hspace=0.35)

    # 1) Solo learning rate
    plot_lr(axs[0], step, df["lr"])

    # 2) Errore globale
    plot_pair(axs[1], step, df["rmse_trn"], df["rmse_val"],
              "Errore globale", "RMSE", args.smooth, args.log)

    # 3) Energie
    plot_pair(axs[2], step, df["rmse_e_trn"], df["rmse_e_val"],
              "Energie (RMSE_e)", "RMSE energia", args.smooth, args.log)

    # 4) Forze
    plot_pair(axs[3], step, df["rmse_f_trn"], df["rmse_f_val"],
              "Forze (RMSE_f)", "RMSE forze", args.smooth, args.log)

    # 5) Viriale
    plot_pair(axs[4], step, df["rmse_v_trn"], df["rmse_v_val"],
              "Viriale (RMSE_v)", "RMSE viriale", args.smooth, args.log)

    axs[-1].set_xlabel("Step di training")
    fig.suptitle("Metriche di training vs step", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(args.out, dpi=300)
    if args.show:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    main()

