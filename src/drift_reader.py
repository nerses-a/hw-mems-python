# drift_reader.py
import numpy as np

def read_drift_file(filename):
    """
    Считывает файл «Дрейф.dat» с двумя столбцами:
      time (с), rate (°/с)

    Возвращает:
      times (np.ndarray), rates (np.ndarray).
    """
    times, rates = [], []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
                r = float(parts[1])
            except ValueError:
                continue
            times.append(t)
            rates.append(r)
    return np.array(times), np.array(rates)

