# reader.py
import numpy as np

def read_first_three_columns(filename):
    times = []
    rates = []
    gyros = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                t = float(parts[0])
                rate = float(parts[1])
                gyr = float(parts[2])
            except ValueError:
                continue
            times.append(t)
            rates.append(rate)
            gyros.append(gyr)

    return np.array(times), np.array(rates), np.array(gyros)
