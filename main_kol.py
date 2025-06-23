import numpy as np
import matplotlib.pyplot as plt

def read_scale_factor_data(file):
    """
    Читает файл МК.dat и группирует строки по скорости (2-я колонка) без использования itertools.groupby.
    Возвращает список пар (velocity, voltage_array), отсортированный по velocity.
    """
    with open(file, mode="r", encoding="utf-8") as f:
        f.readline()  # пропустить строку заголовка
        f.readline()  # пропустить строку с размерностями
        lines = [line.strip() for line in f
                 if line.strip() and not line.lstrip().startswith("#")]

    # Построим словарь: ключ = скорость, значение = список напряжений
    data_dict = {}
    for line in lines:
        parts = list(map(float, line.split()))
        velocity = parts[1]      # вторая колонка
        voltage = parts[2]       # третья колонка (GYR_X_865)
        if velocity not in data_dict:
            data_dict[velocity] = []
        data_dict[velocity].append(voltage)

    # Превратим словарь в список кортежей и вернём
    result = []
    for velocity in sorted(data_dict.keys()):
        voltages_array = np.array(data_dict[velocity])
        result.append((velocity, voltages_array))
    return result

def analyze_scale_factor(data):
    """
    Анализ данных МК.dat:
      • Строит график "Напряжение vs Скорость" с линейной аппроксимацией.
      • Строит график "Нелинейность (%) vs Скорость".
      • Вычисляет и печатает:
        - масштабный коэффициент (scale_factor, В/(°/с)),
        - смещение нуля (zero_offset_volts, В),
        - среднюю нелинейность (%),
        - несимметричность (%),
        - интегральное смещение (%).
    Возвращает: (scale_factor, zero_offset_volts).
    """
    velocities = []
    mean_voltages = []
    for velocity, voltages in data:
        velocities.append(velocity)
        mean_voltages.append(np.mean(voltages))

    X = np.array(velocities)       # скорости (°/с)
    Y = np.array(mean_voltages)    # средние напряжения (В)

    coeffs = np.polyfit(X, Y, 1)
    poly = np.poly1d(coeffs)
    predicted = poly(X)

    scale_factor = coeffs[0]        # В/(°/с)
    zero_offset_volts = coeffs[1]     # смещение нуля в В

    # Нелинейность: |X - (Y - b)/m| / (max|X| * 100)
    predicted_vel = (Y - coeffs[1]) / coeffs[0]
    dev = np.abs(X - predicted_vel)
    nonlinearity = (dev / (np.max(X) - np.min(X))) * 100 # [%]
    mean_nonlinearity = (np.mean(dev) / (np.max(X) - np.min(X))) * 100 # [%]

    # Несимметричность (%)
    if 0 in velocities:
        zero_index = velocities.index(0)
        if 0 < zero_index < len(X) - 1:
            p_coef = np.polyfit(X[zero_index:], Y[zero_index:], 1)[0]
            l_coef = np.polyfit(X[:zero_index+1], Y[:zero_index+1], 1)[0]
            asymmetry = np.abs(np.abs(p_coef) - np.abs(l_coef)) / ((np.abs(p_coef) + np.abs(l_coef)) / 2) * 100
        else:
            asymmetry = np.nan
    else:
        asymmetry = np.nan

    # === График: "Напряжение vs Скорость" ===
    plt.figure(figsize=(9, 5))
    plt.plot(X, Y, 'o', label="Измеренные точки")
    plt.plot(X, predicted, '-', label="Аппроксимация")
    plt.xlabel("Скорость (°/с)")
    plt.ylabel("Напряжение (В)")
    plt.title("Масштабный коэффициент")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === График: "Нелинейность (%) vs Скорость" ===
    plt.figure(figsize=(8, 4))
    plt.plot(X, nonlinearity, 'o-')
    plt.xlabel("Скорость (°/с)")
    plt.ylabel("Нелинейность (%)")
    plt.title("Нелинейность масштабного коэффициента")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Вывод результатов для МК.dat ===
    print("=== Результаты анализа МК.dat ===")
    print(f"Масштабный коэффициент: {scale_factor:.6f} В/(°/с)")
    print(f"Смещение нуля: {zero_offset_volts:.6f} В")
    print(f"Нелинейность: {mean_nonlinearity:.4f} %")
    print(f"Несимметричность: {asymmetry:.4f} %")

    return scale_factor, zero_offset_volts

def read_drift_data(file):
    """
    Читает файл Дрейф.dat, возвращает два массива:
      time (в секундах) и voltage (в вольтах, канал GYR_X_865).
    """
    with open(file, mode="r", encoding="utf-8") as f:
        f.readline()  # пропустить заголовок
        f.readline()  # пропустить строку с единицами
        lines = [line.strip() for line in f
                 if line.strip() and not line.lstrip().startswith("#")]
    data = np.array([list(map(float, line.split())) for line in lines])
    time = data[:, 0]
    voltage = data[:, 1]  # GYR_X_865
    return time, voltage

def polyfit_general(x, y, powers):
    """
    Аппроксимация зависимостей x, y по модели: y ≈ Σ a_i * x^powers[i].
    Возвращает массив коэффициентов [a_0, a_1, ..., a_n].
    """
    x = np.asarray(x)
    y = np.asarray(y)
    A = np.vstack([x**p for p in powers]).T
    ATA = A.T @ A
    ATy = A.T @ y
    coeffs = np.linalg.solve(ATA, ATy)
    return coeffs

def analyze_drift(time, voltage, scale_factor):
    """
    Анализ данных дрейфа:
      • Строит график "Сырые данные дрейфа" (напряжение в Вольтах) + прямая аппроксимации (в В/с).
      • Вычисляет и печатает:
        – смещение нуля (в В и в °/с),
        – тренд (°/ч/ч),
      • Строит график "Девиация Аллана" и аппроксимирует:
        – ARW (τ^{-0.5}), Bias Instability и RRW (τ^{0.5}),
        – а также модель шумов (черная пунктирная линия).
    """
    # --- Смещение нуля ---
    bias_volts = np.mean(voltage)
    bias_dps = bias_volts / scale_factor

    # --- Перевод всей последовательности в °/с ---
    gyr_dps = voltage / scale_factor

    # --- Аппроксимация "Напряжение vs Время" (для тренда в Вольт/с) ---
    coeffs_v = np.polyfit(time, voltage, 1)
    slope_v = coeffs_v[0]      # В/с
    intercept_v = coeffs_v[1]  # В

    # Переводим наклон из В/с в °/с²: slope_v * (1/scale_factor)
    # Затем в °/ч/ч: умножить на (3600^2)
    trend_dphph = slope_v * (1.0 / scale_factor) * (3600**2)

    # === График: "Сырые данные дрейфа" (Вольты) + тренд ===
    plt.figure(figsize=(8, 4))
    plt.plot(time, voltage, color='tab:blue', linewidth=0.5, label="Сырые данные (В)")
    plt.plot(time, slope_v * time + intercept_v, color='tab:orange', linestyle='--',
             label=f"Тренд: y={slope_v:.3e}·x + {intercept_v:.3f} В")
    plt.xlabel("Время, с")
    plt.ylabel("Напряжение (В)")
    plt.title("Дрейф гироскопа")
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

    # === Вывод результатов для Дрейф.dat ===
    print("=== Результаты анализа Дрейф.dat ===")
    print(f"Смещение нуля: {bias_volts:.6f} В => {bias_dps:.6f} °/с")
    print(f"Тренд (°/ч/ч): {trend_dphph:.2f}\n")

    # === Вычисление Allan Deviation ===
    def allan_deviation(signal, fs):
        """
        Возвращает (taus, adev) для входного сигнала (signal) с частотой дискретизации fs.
        """
        n = len(signal)
        max_m = int(np.floor(n / 2))
        m_vals = np.logspace(0, np.log10(max_m), num=100, dtype=int)
        m_vals = np.unique(m_vals)

        taus = m_vals / fs
        adevs = []
        for m in m_vals:
            means = np.array([np.mean(signal[i:i+m]) for i in range(0, n - m + 1, m)])
            if len(means) < 2:
                break
            diffs = np.diff(means)
            adev = np.sqrt(0.5 * np.mean(diffs**2))
            adevs.append(adev)
        return taus[:len(adevs)], np.array(adevs)

    # np.diff(time) — массив разностей между соседними временами
    # np.mean(...) — средний временной шаг (в секундах)
    fs = 1 / np.mean(np.diff(time))
    taus, adevs = allan_deviation(gyr_dps, fs)

    # === Вычисление и аппроксимация параметров шума ===
    min_idx = np.nanargmin(adevs)
    bias_inst = adevs[min_idx]
    tau_min = taus[min_idx]

    # ARW: τ < tau_min
    mask_white = taus < tau_min
    p_white = np.polyfit(np.log10(taus[mask_white]), np.log10(adevs[mask_white]), 1)
    ARW = (10**p_white[1]) * 60  # перевод в °/√ч (√3600 = 60)

    # RRW: τ > tau_min
    mask_rrw = taus > tau_min
    p_rrw = np.polyfit(np.log10(taus[mask_rrw]), np.log10(adevs[mask_rrw]), 1)
    RRW = (10**p_rrw[1]) * 3600  # перевод в °/√(ч³)

    # Модель (сумма шумов) степеней [-2, -1, 0, 1, 2]
    powers = [-2, -1, 0, 1, 2]
    coeffs_model = polyfit_general(taus, adevs, powers)
    model = sum(c * taus**p for c, p in zip(coeffs_model, powers))

    # === График: Allan Deviation с аппроксимациями ===
    plt.figure(figsize=(10, 6))
    plt.loglog(taus, adevs, color="blue", label="Данные")
    plt.scatter(tau_min, bias_inst, color="red", s=50, label="Bias Instability")

    # ARW (τ^{-0.5})
    tau_fit_white = taus[mask_white]
    fit_white = 10**p_white[1] * tau_fit_white**p_white[0]
    plt.loglog(tau_fit_white, fit_white, color="green", label="ARW (τ^{-0.5})")

    # RRW (τ^{0.5})
    tau_fit_rrw = taus[mask_rrw]
    fit_rrw = 10**p_rrw[1] * tau_fit_rrw**p_rrw[0]
    plt.loglog(tau_fit_rrw, fit_rrw, color="magenta", label="RRW (τ^{0.5})")

    # Модель (сумма шумов)
    plt.loglog(taus, model, '--', color="black", label="Модель (сумма шумов)")

    plt.xlabel("tau [сек]")
    plt.ylabel("sigma(tau)")
    plt.title("Анализ девиации Аллана")
    plt.grid(which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Вывод шумовых параметров ===
    print(f"ARW (°/√ч): {ARW:.4f}")
    print(f"Bias Instability (°/ч): {bias_inst:.4f}")
    print(f"RRW (°/ч/√ч): {RRW:.4f}")

if __name__ == "__main__":
    # Обработка МК.dat
    sf, zero_offset = analyze_scale_factor(read_scale_factor_data("МК.dat"))

    # Обработка Дрейф.dat
    time, voltage = read_drift_data("Дрейф.dat")
    analyze_drift(time, voltage, sf)

