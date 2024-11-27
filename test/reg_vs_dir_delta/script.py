import os
from scipy.stats import wilcoxon, linregress
import csv
import numpy as np

OUT_FILE = 'out.txt'

if os.path.exists(OUT_FILE):
    os.remove(OUT_FILE)

with open('results.csv', newline='') as csvfile:
    rows = []

    for line in csv.reader(csvfile, delimiter=';', quotechar='|'):
        rows.append(line)

    data = []

    for c in range(1, len(rows[0])):
        i = 0
        
        for r in range(1, len(rows)):
            data.append(float(rows[r][c]))

            i += 1

            if i % 5 == 0:
                #print(data)
                reference_value = np.mean(data)
                #stat, p = wilcoxon(data - reference_value)

                #print(f"{rows[0][c]} & {rows[i][0]}> Test Statistic: {stat}, p-value: {p}")

                t = np.arange(len(data))
                slope, intercept, r_value, p_value, std_err = linregress(t, data)

                sdev = np.std(data)
                #np.std()

                out = open(OUT_FILE, "a")
                out.write(f"{rows[0][c]} \t {rows[i][0]} \t {slope}\n")
                out.close()

                print(f"{rows[0][c]} & {rows[i][0]}> \t slope={slope} \t slope test={np.abs(slope) > 0.01}")

                data = []