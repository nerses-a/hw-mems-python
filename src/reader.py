# reader.py
import numpy as np

def read_first_three_columns(filename):
    """
    Считывает файл «МК.dat» с тремя столбцами:
      time (с), rate (°/с), gyro output (В)

    Возвращает:
      times (np.ndarray) — время, 
      omega (np.ndarray) — входная угловая скорость,
      U (np.ndarray)     — выход гироскопа (В).
    """
    times, omega, U = [], [], []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                t = float(parts[0])
                w = float(parts[1])
                v = float(parts[2])
            except ValueError:
                continue
            times.append(t)
            omega.append(w)
            U.append(v)
    return np.array(times), np.array(omega), np.array(U)

