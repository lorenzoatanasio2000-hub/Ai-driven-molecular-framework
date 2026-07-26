import re
import statistics

temps = []
densities = []
totengs = []

with open("lmp_output.log") as f:
    capture = False
    for line in f:
        # find the head of the table
        if re.match(r"\s*Step\s+Temp\s+PotEng", line):
            capture = True
            continue
        if capture:
            parts = line.strip().split()
            # table explanation
            if len(parts) == 7 and parts[0].isdigit():
                temp = float(parts[1])
                toteng = float(parts[4])
                density = float(parts[5])
                temps.append(temp)
                totengs.append(toteng)
                densities.append(density)

# mean value 
mean_temp = statistics.mean(temps)
mean_density = statistics.mean(densities)
mean_toteng = statistics.mean(totengs)

print(f"Media Temperatura: {mean_temp:.3f} K")
print(f"Media Densità: {mean_density:.6f} g/cm^3")
print(f"Media Energia Totale: {mean_toteng:.6f} eV")

