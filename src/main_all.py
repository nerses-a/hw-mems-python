import matplotlib.pyplot as plt
import numpy as np
import math


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

def calc_scale_metrics(omega, U):
    '''
    Расчет метрических параметров по данным omega (°/с) и U (В) с возвратом списка:
      [
        SF [mV/(°/с)],
        LinearityError [%],
        Asymmetry [%],
        ZeroOffset [mV],
        a_raw,
        b_raw_mV
      ]

    Формулы:
      U_mV = U · 1000
      a_raw, b_raw = polyfit(omega, U_mV, 1)
      SF = a_raw  # масштабный коэффициент в мВ/(°/с)

      uniq = unique(omega)
      mean_U = [mean(U_mV[omega==w]) for w in uniq]
      U_FS = max(mean_U) - min(mean_U)

      NL_i = 100 · (mean_U - (SF·uniq + b_raw)) / U_FS
      LinearityError = mean(|NL_i|)

      Asymmetry: вычисление по участкам вокруг нуля
      ZeroOffset = b_raw  # в мВ
    '''
     # перевод в мВ
    U_mV = U * 1e3

    # аппроксимация по всем точкам
    a_raw, b = np.polyfit(omega, U_mV, 1)
    SF = abs(a_raw)

    # средние выходные для каждой уникальной скорости
    uniq = np.unique(omega)
    mean_U = np.array([U_mV[omega == w].mean() for w in uniq])

    # полный диапазон средних выходных (мВ)
    U_FS = mean_U.max() - mean_U.min()


     # нелинейность: макс отклонение средних от линии регрессии
    deviations = np.abs(mean_U - (a_raw * uniq + b))
    nonlin = (100.0 * deviations / U_FS)

    # асимметрия: по индексам вокруг нуля
    velocities = uniq.tolist()
    if 0 in velocities:
        zero_index = velocities.index(0)
        if 0 < zero_index < len(uniq) - 1:
            a_R = np.polyfit(uniq[zero_index:], mean_U[zero_index:], 1)[0]
            a_L = np.polyfit(uniq[:zero_index+1], mean_U[:zero_index+1], 1)[0]
            asym = np.abs(np.abs(a_R) - np.abs(a_L)) / ((np.abs(a_R) + np.abs(a_L)) / 2) * 100
        else:
            asym = np.nan
    else:
        asym = np.nan


     # смещение нуля (°/с)
    zero_offset = b

    return {
        'SF': SF,
        'Linearity_%': nonlin,
        'Asymmetry_%SF': asym,
        'ZeroOffset_°/s': zero_offset,
        'a_raw': a_raw,
        'b': b
    }
# ---- Gauss solver for weighted least squares ----
def solve_gauss(matrix, vector):
    n = len(matrix)
    for col in range(n):
        # find pivot
        max_row = max(range(col, n), key=lambda r: abs(matrix[r][col]))
        matrix[col], matrix[max_row] = matrix[max_row], matrix[col]
        vector[col], vector[max_row] = vector[max_row], vector[col]
        pivot = matrix[col][col]
        # normalize
        for c in range(col, n):
            matrix[col][c] /= pivot
        vector[col] /= pivot
        # eliminate
        for r in range(n):
            if r != col and matrix[r][col] != 0:
                factor = matrix[r][col]
                for c in range(col, n):
                    matrix[r][c] -= factor * matrix[col][c]
                vector[r] -= factor * vector[col]
    return vector

# -- Drift & noise metrics --
def calc_drift_metrics(times, rates):
    Z = np.mean(rates)
    a, b = np.polyfit(times, rates, 1)
    trend = a * (3600**2)
    fs = 1.0 / (times[1] - times[0])

    def allan_deviation(data, fs):
        n = len(data)
        m_vals = []
        m = 1
        while m <= n//2:
            m_vals.append(m)
            m = int(math.ceil(m * 1.2))
        taus, adevs = [], []
        for m in m_vals:
            # block means
            means = [np.mean(data[i:i+m]) for i in range(0, n-m+1, m)]
            if len(means) < 2:
                break
            diffs = np.diff(means)
            sigma2 = 0.5 * np.mean(diffs**2)
            taus.append(m/fs)
            adevs.append(math.sqrt(sigma2))
        return np.array(taus), np.array(adevs)

    taus, adevs = allan_deviation(rates, fs)
    # ARW, BI, RRW same as before
    log_t = np.log10(taus)
    log_a = np.log10(adevs)
    N = len(taus)
    n_arw = max(2, int(0.33 * N))
    _, b_arw = np.polyfit(log_t[:n_arw], log_a[:n_arw], 1)
    ARW = 10**b_arw * math.sqrt(3600)
    BI = adevs.min() * 3600
    start_rrw = max(0, N - max(2, int(0.1 * N)))
    _, b_rrw = np.polyfit(log_t[start_rrw:], log_a[start_rrw:], 1)
    RRW = 10**b_rrw * math.sqrt(3) * (3600**1.5)

    return {
        'ZeroOffset': Z,
        'Trend': trend,
        'taus': taus,
        'adevs': adevs,
        'ARW': ARW,
        'BI': BI,
        'RRW': RRW
    }

# ---- Polynomial fit & approximation like in reference code ----
def polyfit_allan(tau, adev, powers):
    # log-scale and weights
    log_tau = [math.log10(t) for t in tau]
    mean_log_tau = sum(log_tau)/len(log_tau)
    delta_log = [log_tau[0]] + [log_tau[i]-log_tau[i-1] for i in range(1, len(log_tau))]
    window = [math.exp(-((lt-mean_log_tau)**2)/(2*1.0**2)) for lt in log_tau]
    weights = [dl*w for dl, w in zip(delta_log, window)]
    # feature matrix
    A = [[t**p for p in powers] for t in tau]
    # build W, ATWA, ATWadev
    n = len(weights)
    W = [[0]*n for _ in range(n)]
    for i in range(n): W[i][i] = weights[i]
    m = len(powers)
    ATWA = [[0]*m for _ in range(m)]
    ATWadev = [0]*m
    for i in range(m):
        for j in range(m):
            for k in range(len(tau)):
                ATWA[i][j] += A[k][i]*W[k][k]*A[k][j]
    for i in range(m):
        for k in range(len(tau)):
            ATWadev[i] += A[k][i]*W[k][k]*adev[k]
    coeffs = solve_gauss(ATWA, ATWadev)
    return coeffs

# returns coeffs, model, tau_filtered
def plot_allan_approximation(tau, adev, powers):
    t_min, t_max = 0.33, 3500
    mask = [(t>t_min and t<t_max) for t in tau]
    tau_f = [t for t,m in zip(tau, mask) if m]
    adev_f = [a*a for a,m in zip(adev, mask) if m]
    coeffs = polyfit_allan(tau_f, adev_f, powers)
    model = [sum(c*t**p for c,p in zip(coeffs, powers)) for t in tau_f]
    return coeffs, model, tau_f

def analyze_scale(file_scale, do_plot=False):
    t, omega, U = read_first_three_columns(file_scale)
    M = calc_scale_metrics(omega, U)
    uniq = np.unique(omega)
    print("=== Scale Metrics ===")
    print(f"Scale Factor:    {M['SF']:.4f} mV/(°/s)")
    print(f"Linearity Err:   {M['Linearity_%'].max():.4f} %FS")
    print(f"Asymmetry Err:   {M['Asymmetry_%SF']:.4f} %SF")
    print(f"Zero Offset:     {M['ZeroOffset_°/s']:.4f} °/s")
    if do_plot:
        U_mV = U * 1e3
        a, b = M['a_raw'], M['b']
        ω_lin = np.linspace(omega.min(), omega.max(), 200)
        plt.figure()
        plt.scatter(omega, U_mV, s=10, label='raw')
        plt.plot(ω_lin, a*ω_lin + b, 'r-', label='fit')
        plt.title("Scale Factor Fit"); plt.legend();
        plt.grid(True)

        plt.figure()
        plt.plot(uniq , M['Linearity_%'], 'r-', label='fit')
        plt.title("Linearity %"); plt.legend();
        plt.grid(True)

        plt.show(block=False)

def analyze_drift(file_drift, SF, do_plot=False):
    # Чтение файла дрейфа
    times, voltages = read_drift_file(file_drift)
    # Конвертация напряжения в скорость (°/s)
    rates = np.array(voltages) / (SF / 1e3)

    # Вычисление метрик дрейфа и шума
    metrics = calc_drift_metrics(np.array(times), rates)
    print("=== Drift Metrics ===")
    print(f"Zero Offset: {metrics['ZeroOffset']:.6f} °/s")
    print(f"Trend:       {metrics['Trend']:.4f} °/h²")
    print(f"ARW:         {metrics['ARW']:.6f} °/√h")
    print(f"BI:          {metrics['BI']:.6f} °/h")
    print(f"RRW:         {metrics['RRW']:.6f} °/h³/²")

    if do_plot:
        # Plot drift signal
        plt.figure()
        plt.plot(times, rates - metrics['ZeroOffset'], label='Drift Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Rate (°/s)')
        plt.title('Gyro Drift over Time')
        plt.grid(True)
        plt.legend()
        plt.show(block=False)

        # Plot Allan deviation and polynomial approximation
        taus = metrics['taus']
        adevs = metrics['adevs']
        # raw Allan deviation
        plt.figure()
        plt.loglog(taus, adevs*3600, 'bo-', label='Allan Deviation')
        # polynomial fit
        powers = [-2, -1, 0, 1]
        coeffs, model_vals, tau_f = plot_allan_approximation(taus, adevs, powers)
        plt.loglog(tau_f, [math.sqrt(v)*3600 for v in model_vals], 'r--', label='Approximation')
        plt.xlabel('τ (s)')
        plt.ylabel('σ(τ) (°/h)')
        plt.title('Allan Deviation & Approximation')
        plt.legend()
        plt.grid(True, which='both', ls='--')
        plt.show()

if __name__ == "__main__":
    scale_file = "МК.dat"
    drift_file = "Дрейф.dat"
    show_plots = True  # Флаг для отображения графиков (True/False)

    SF = None
    if scale_file:
        analyze_scale(scale_file, do_plot=show_plots)
        # Получаем масштабный коэффициент из анализа
        # Здесь упрощённо: читаем файл и пересчитываем SF
        _, omega, U = read_first_three_columns(scale_file)
        M = calc_scale_metrics(omega, U)
        SF = M['SF']

    if drift_file:
        if SF is None:
            print("Warning: scale factor not provided, using SF=1.0 mV/(°/s)")
            SF = 1.0
        analyze_drift(drift_file, SF, do_plot=show_plots)

    if show_plots:
        input("Press Enter to exit...")
