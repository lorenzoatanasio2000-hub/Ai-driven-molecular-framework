filename = "lmp_output.log"

densities = []
density_index = None
header_found = False

with open(filename) as f:
    for line in f:
        parts = line.split()

        # Trova la riga di intestazione
        if "Density" in parts and not header_found:
            density_index = parts.index("Density")
            header_found = True
            continue

        # Se abbiamo trovato l'intestazione, leggiamo i dati numerici
        if header_found:
            try:
                value = float(parts[density_index])
                densities.append(value)
            except (ValueError, IndexError):
                # se la riga non è numerica, ci fermiamo
                break

if densities:
    media = sum(densities) / len(densities)
    print(f"Media della colonna 'Density': {media}")
else:
    print("⚠️ Nessun valore trovato nella colonna 'Density'")

