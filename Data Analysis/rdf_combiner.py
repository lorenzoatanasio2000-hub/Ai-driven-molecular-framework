#!/usr/bin/env python3
"""
combine_rdf.py

Combina più file RDF (r, g(r)) in un unico grafico PNG.
Funziona con file separati da ';' come quelli generati da TraVis:
Distance / pm ; g(r) ; Integral

Le curve saranno etichettate automaticamente con la parte finale del nome file
che indica il tempo, ad esempio 'rdf_Li_Li_2ns.csv' -> '2ns'.

Uso:
    python3 combine_rdf.py rdf_Li_Li_2ns.csv rdf_Li_Li_5ns.csv --output combined_rdf.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_rdf(file):
    """Carica RDF da file separato da ';', ritorna r in Å e g(r)."""
    try:
        df = pd.read_csv(file, sep=';', comment='#', header=None)
        if df.shape[1] < 2:
            raise ValueError(f"{file} non ha almeno due colonne")
        r = df.iloc[:,0].values / 100.0  # Converti pm -> Å
        g = df.iloc[:,1].values
        return r, g
    except Exception as e:
        print(f"Errore caricamento {file}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Combina più RDF in un unico grafico")
    parser.add_argument("files", nargs="+", help="File RDF (r, g(r))")
    parser.add_argument("--output", default="combined_rdf.png", help="Nome file PNG di output")
    parser.add_argument("--colors", nargs="+", help="Colori per ogni curva (opzionale)")
    parser.add_argument("--show", action="store_true", help="Mostra il grafico a video")
    parser.add_argument("--smooth", type=int, default=1, help="Media mobile (opzionale)")
    args = parser.parse_args()

    plt.figure(figsize=(8,6))

    for i, file in enumerate(args.files):
        r, g = load_rdf(file)
        if r is None or g is None:
            continue

        # Applica smoothing se richiesto
        if args.smooth > 1:
            g = pd.Series(g).rolling(window=args.smooth, center=True, min_periods=1).mean().values

        # Colore opzionale
        color = None
        if args.colors and i < len(args.colors):
            color = args.colors[i]

        # Estrai etichetta dal nome file: ultima parte prima di .csv
        label = os.path.basename(file).split("_")[-1].replace(".csv","")

        # Traccia la curva
        plt.plot(r, g, label=label, color=color)

    # Formattazione grafico
    plt.xlabel("r (Å)", fontsize=14)
    plt.ylabel("g(r)", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc="upper right")
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"✅ Grafico salvato in: {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    main()

