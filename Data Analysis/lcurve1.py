#!/usr/bin/env python3
"""
plot_training_split.py

Crea due file PNG dai dati di training:

1) learning_global.png
   - 2 subplot affiancati:
     - Learning rate
     - Errore globale

2) errors_EFV.png
   - 3 subplot su 2 righe:
     - Prima riga: Energie, Forze
     - Seconda riga: Viriale (centrato)
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sys

def read_data(path):
    if path == "-":
        df = pd.read_csv(sys.stdin, comment="#", delim_whitespace=True, header=None)
    else:
        df = pd.read_csv(path, comment="#", delim_whitespace=True, header=None)
    cols = ["step","rmse_val","rmse_trn","rmse_e_val","rmse_e_trn",
            "rmse_f_val","rmse_f_trn","rmse_v_val","rmse_v_trn","lr"]
    if df.shape[1] < len(cols):
        raise ValueError(f"Il file ha {df.shape[1]} colonne ma mi aspettavo almeno {len(cols)}.")
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
    ax.plot(step, ytr, linewidth=1, label="Train")
    ax.plot(step, yvl, linewidth=1, linestyle="-", label="Validation")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="-", alpha=0.35)
    if logscale:
        ax.set_yscale("log")
    ax.legend(loc="upper right", frameon=False, fontsize="small")

def plot_lr(ax, step, lr):
    ax.plot(step, lr, linewidth=1, linestyle="-", color="gray", label="Learning rate")
    ax.set_title("Learning rate")
    ax.set_ylabel("Learning rate")
    ax.grid(True, linestyle="-", alpha=0.35)
    ax.legend(loc="upper right", frameon=False, fontsize="small")

def main():
    p = argparse.ArgumentParser(description="Split plot: LR+global, E/F/V")
    p.add_argument("file", help="file dati (o - per stdin)")
    p.add_argument("--smooth", type=int, default=1, help="rolling average window")
    p.add_argument("--log", action="store_true", help="scala log per asse y")
    p.add_argument("--show", action="store_true", help="mostra grafico a video")
    args = p.parse_args()

    df = read_data(args.file)
    step = df["step"]

    # --- Primo file: Learning rate + Errore globale ---
    fig1, axs1 = plt.subplots(1, 2, figsize=(14,5), sharex=True)
    plot_lr(axs1[0], step, df["lr"])
    plot_pair(axs1[1], step, df["rmse_trn"], df["rmse_val"], "Errore globale", "RMSE", args.smooth, args.log)

    # Aumenta dimensione label, titolo e numeri
    for ax in axs1:
        ax.set_xlabel("Step di training", fontsize=14)
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.title.set_fontsize(16)
        # Aggiorna legenda in modo compatibile
        leg = ax.get_legend()
        if leg:
            for text in leg.get_texts():
                text.set_fontsize(12)

    fig1.tight_layout()
    fig1.savefig("learning_global.png", dpi=300)
    if args.show:
        plt.show()
    else:
        plt.close(fig1)

    # --- Secondo file: Energie, Forze, Viriale ---
    fig2 = plt.figure(figsize=(14,9))
    import matplotlib.gridspec as gridspec
    gs = fig2.add_gridspec(2, 2, height_ratios=[1,1], width_ratios=[1,1], hspace=0.35, wspace=0.3)
    ax0 = fig2.add_subplot(gs[0,0])  # Energie
    ax1 = fig2.add_subplot(gs[0,1])  # Forze
    ax2 = fig2.add_subplot(gs[1, :]) # Viriale, span due colonne per centrarlo

    plot_pair(ax0, step, df["rmse_e_trn"], df["rmse_e_val"], "Energie (RMSE_e)", "RMSE energia", args.smooth, args.log)
    plot_pair(ax1, step, df["rmse_f_trn"], df["rmse_f_val"], "Forze (RMSE_f)", "RMSE forze", args.smooth, args.log)
    plot_pair(ax2, step, df["rmse_v_trn"], df["rmse_v_val"], "Viriale (RMSE_v)", "RMSE viriale", args.smooth, args.log)

    for ax in [ax0, ax1, ax2]:
        ax.set_xlabel("Step di training")

    fig2.tight_layout()
    fig2.savefig("errors_EFV.png", dpi=300)
    if args.show:
        plt.show()
    else:
        plt.close(fig2)

if __name__ == "__main__":
    main()

