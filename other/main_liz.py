# импорт библиотек
import matplotlib.pyplot as plt
import math
import allantools as at
import numpy as np

#----ФУНКЦИИ------ НАЧАЛО

# Решение СЛАУ методом Гаусса
def solve_gauss(matrix, vector):
    n = len(matrix)
    # цикл по столбцам матрицы
    for col in range(n):
        # Поиск максимального элемента в столбце
        max_row = max(range(col, n), key=lambda r: abs(matrix[r][col]))
        # Перестановка строк для выбора ведущего элемента
        # Если ведущий элемент не на диагонали, текущая строка col
        # меняется местами со строкой max_row
        # Аналогично переставляются элементы вектора правых частей.
        matrix[col], matrix[max_row] = matrix[max_row], matrix[col]
        vector[col], vector[max_row] = vector[max_row], vector[col]

        # Нормализация текущей строки
        # pivot — ведущий элемент (диагональный элемент после перестановки).
        pivot = matrix[col][col]
        for c in range(col, n):
            matrix[col][c] /= pivot
        vector[col] /= pivot

        # Исключение переменной из остальных строк
        for r in range(n):
            if r != col and matrix[r][col] != 0:
                factor = matrix[r][col]
                for c in range(col, n):
                    matrix[r][c] -= factor * matrix[col][c]
                vector[r] -= factor * vector[col]

    return vector


#ПОЛИНОМ АЛЛАНА и график
# на вход подаём время, уровень шума и коэффициенты -2 -1 0 1
def polyfit_allan(tau, adev, powers):
    # Переводим τ в логарифмическую шкалу
    # для устойчивого взвешивания точек (Allan deviation часто строится в log-log).
    log_tau = [math.log10(t) for t in tau]
    mean_log_tau = sum(log_tau) / len(log_tau)
    #delta_log — измеряет "ширину" окна между соседними τ по логарифму.
    delta_log = [log_tau[0]] + [log_tau[i] - log_tau[i - 1] for i in range(1, len(log_tau))]
    #window — обеспечивает гауссово взвешивание вокруг
    # центра log(τ), чтобы крайние точки в меньшей степени влияли на аппроксимацию.
    # чем ближе к центру графика, тем больше вес, чем дальше от центра- тем меньше вес
    window = [math.exp(-((lt - mean_log_tau) ** 2) / (2 * 1.0 ** 2)) for lt in log_tau]
    #weights — итоговые веса для каждой точки: произведение
    # "ширины" лог окна и гауссового окна относительно центра.
    weights = [dl * w for dl, w in zip(delta_log, window)]

    # Формирование матриц для взвешенного МНК
    # Строим матрицу фичей: каждая строка — это значения τ в разных степенях
    # (для конкретной τ).
    A = [[t ** p for p in powers] for t in tau]

    # Создание диагональной матрицы весов
    W = [[0] * len(weights) for _ in range(len(weights))]
    for i in range(len(weights)):
        W[i][i] = weights[i]
    # Подготовка к решению взвешенной системы линейных уравнений
    ATWA = [[0] * len(powers) for _ in range(len(powers))]
    ATWadev = [0] * len(powers)

    for i in range(len(powers)):
        for j in range(len(powers)):
            for k in range(len(tau)):
                ATWA[i][j] += A[k][i] * W[k][k] * A[k][j]

    for i in range(len(powers)):
        for k in range(len(tau)):
            ATWadev[i] += A[k][i] * W[k][k] * adev[k]
    # Решаем систему методом Гаусса. На выходе — коэффициенты (Q, N, B, K).
    coeffs = solve_gauss(ATWA, ATWadev)
    return coeffs




#ФУНКЦИЯ ДЛЯ ПОЛУЧЕНИЯ КОЭФФИЦИЕНТА ARW, BI, RRW  и графика
def plot_allan_approximation(tau, adev, powers):
    # задаются границы интервалов усреднения Allan Deviation (обычно в секундах).
    t_min = 0.33
    t_max = 500
    # задаем маску, в которой содеражся нужные t принадлежащие выще заданным границам
    mask = [(t > t_min) and (t < t_max) for t in tau] # Оставляются только те τ, которые попадают в заданный диапазон.
    tau_filtered = [t for t, m in zip(tau, mask) if m] # Аналогично для Allan deviation: оставляем только релевантные
    # значения и возводим их в квадрат (т.е. переходим от стандартного отклонения к дисперсии).
    adev_filtered = [a * a for a, m in zip(adev, mask) if m]
    #Здесь происходит аппроксимация дисперсии Аллана (σ²) суммой законов разных шумов.
    #Список powers задаёт степени τ ([-2, -1, 0, 1] — это Q, N, B, K).
    coeffs = polyfit_allan(tau_filtered, adev_filtered, powers)
    # Вычисляется значение аппроксимирующей функции (модели) дисперсии Аллана в каждой точке tau_filtered.
    # модель для построения графика
    model = [sum(c * t ** p for c, p in zip(coeffs, powers)) for t in tau_filtered]
    return coeffs, model, tau_filtered


# девиация Аллана
def allan_deviation(data, fs):
    n = len(data) # Количество точек данных
    max_m = n // 2  # Максимальный размер интервала (половина данных)
    m_vals = [] # сюда будем записывать размеры окон (m)
    # Чтобы охватить все возможные интервалы усреднения τ
    # от минимального (1/fs) до максимального ((n/2)/fs).
    # Генерируем логарифмические интервалы
    m = 1
    while m <= max_m:
        m_vals.append(m)
        m = int(math.ceil(m * 1.2))  # увеличиваем примерно на 20%

    # перевод в секунды
    taus = [m / fs for m in m_vals]

    # основной цикл по интервалам m
    adevs = []

    for m in m_vals:
        # Разбиваем данные на блоки длины m и вычисляем их среднее
        means = []
        for i in range(0, n - m + 1, m):
            chunk = data[i:i + m]
            mean = sum(chunk) / m
            means.append(mean)

        # проверка числа блоков
        if len(means) < 2:
            break  # Нужно минимум два средних, чтобы посчитать разницу

        # Вычисляем разности соседних средних
        diffs = []
        for i in range(len(means) - 1):
            diffs.append(means[i + 1] - means[i])

        # Расчёт девиации Аллана
        # Квадраты разностей
        squared_diffs = [d ** 2 for d in diffs]
        # Среднее квадратов
        mean_squared_diff = sum(squared_diffs) / len(squared_diffs)
        # Итоговая Allan Deviation
        adev = math.sqrt(0.5 * mean_squared_diff)
        adevs.append(adev)

    return taus[:len(adevs)], adevs

def linear_regression(X, Y):
    # Вычисляем суммы сигналы
    n = len(X)

    sum_x = sum(X)  # сумма значений по Х
    sum_y = sum(Y)  # сумма значений по У
    sum_xy = 0
    for i in range(n):
        sum_xy += X[i] * Y[i] # Сумма произведений пар (x, y)
    sum_x2 = sum(x ** 2 for x in X)  # Сумма квадратов значений Х

    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)  # вычисляем a
    b = (sum_y - a * sum_x) / n  # вычисляем b
    return a, b

def read_table(filename, time_col=0, omega_col=1, voltage_col=2):

    time = []
    omega = []
    voltage = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('time'):
                parts = line.strip().split()
                try:
                    time.append(float(parts[time_col]))
                    omega.append(float(parts[omega_col]))
                    voltage.append(float(parts[voltage_col]))
                except (IndexError, ValueError):
                    continue  # пропускаем заголовки и некорректные строки
    return time, omega, voltage

# делит весь интервал времени на окна по n минут и считаем тренд гироскопа на каждом таком интервале
def mean_in_n_min_windows(time, gyro_rate, n_minutes):
    window_size = n_minutes * 60  # Переводим минуты в секунды
    start_time = min(time)        # Начальное время в данных
    end_time = max(time)          # Конечное время в данных
    trends = []                   # Список для хранения трендов (наклонов) по окнам
    window_centers = []           # Список для хранения центров каждого окна

    current_start = start_time
    while current_start < end_time:
        current_end = current_start + window_size  # Формируем окно нужной длины
        window_times = []
        window_rates = []
        # Собираем значения, попадающие в текущее окно
        for t, g in zip(time, gyro_rate):
            if current_start <= t < current_end:
                window_times.append(t)
                window_rates.append(g)
        # Если в окне есть хотя бы две точки,
        # считаем линейную регрессию - наклон slope
        if len(window_times) >= 2:
            slope, intercept = linear_regression(window_times, window_rates)
            trends.append(slope * 3600 ** 2)
            # Сохраняем наклон (тренд) этого окна по времени
            window_centers.append(current_start + window_size / 2)
        current_start += window_size  # Сдвигаем окно вперед
    return window_centers, trends  # Возвращаем центры окон и соответствующие тренды
# -end


#РАССЧЕТ ТРЕНДА ОТ НАЧАЛА  START
def calc_trend_from_begin_to_minute(time, gyro_rate, minute):
    # Предполагается, что time — список времени в секундах, gyro_rate — значения гироскопа
    start_time = min(time)
    end_time = start_time + minute * 60  # первые две минуты

    # Отбираем точки только из первых двух минут
    t_window = []
    g_window = []
    for t, g in zip(time, gyro_rate):
        if start_time <= t < end_time:
            t_window.append(t)
            g_window.append(g)

    # Считаем линейную регрессию
    trend, intercept = linear_regression(t_window, g_window)
    return t_window, trend * 3600 ** 2
# РАССЧЕТ ТРЕНДА ОТ НАЧАЛА END

def calculate_nonlinearity_by_formula(rate_values, gyro_values, scale_factor, bias):
    """
    Расчет нелинейности по формуле NL_i = 100 × (GYRO_i - (S×RATE_i + B)) / (max(S×RATE+B) - min(S×RATE+B))

    Parameters:
    rate_values: список эталонных угловых скоростей (RATE_i)
    gyro_values: список измеренных значений гироскопа (GYRO_i)
    scale_factor: масштабный коэффициент (S)
    bias: смещение нуля (B)

    Returns:
    nonlinearity_values: список значений нелинейности для каждой точки
    avg_nonlinearity: средняя нелинейность
    """

    # Вычисляем идеальные значения S×RATE_i + B для каждой точки
    ideal_values = [scale_factor * rate + bias for rate in rate_values]

    # Находим максимальное и минимальное идеальные значения
    max_ideal = max(ideal_values)
    min_ideal = min(ideal_values)
    denominator = max_ideal - min_ideal

    # Вычисляем нелинейность для каждой точки по формуле
    nonlinearity_values = []
    for i in range(len(rate_values)):
        gyro_i = gyro_values[i]
        ideal_i = ideal_values[i]
        nl_i = 100 * (gyro_i - ideal_i) / denominator
        nonlinearity_values.append(nl_i)

    # Средняя нелинейность (по модулю)
    avg_nonlinearity = sum(abs(nl) for nl in nonlinearity_values) / len(nonlinearity_values)

    return nonlinearity_values, avg_nonlinearity, ideal_values




# =============================================
# 1. ОБРАБОТКА МАСШТАБНОГО КОЭФФИЦИЕНТА (МК.dat)
# =============================================
print("1. ОБРАБОТКА МАСШТАБНОГО КОЭФФИЦИЕНТА (МК.dat)")
# Словарь для хранения всех сигналов по скоростям
all_voltages = {}
# Загрузка данных МК
_, omega, voltage = read_table('МК.dat', omega_col=1, voltage_col=2)

for i in range(len(omega)):
    if omega[i] not in all_voltages:  # это проверка, если в словаре еще нет значения для этой скорости мы создаем массив, в который запихнем как раз сигнал
        all_voltages[omega[i]] = []  # пустой масссив для определенной скорости.
    all_voltages[omega[i]].append(voltage[i])  # Добавляем сигнал для нужной скорости


velocities = list(all_voltages.keys()) # - это массив который содержит скорости [-50; 0; 50]

mean_voltages = []
for vel in velocities:
    voltages_list = all_voltages[vel]
    mean_value = sum(voltages_list) / len(voltages_list)  # расчет среднего значения сигнала для каждой скорости
    mean_voltages.append(mean_value) # добавление в массив

# для того, чтобы узнать нелинейность значений скорости воспользуемся методом ЛИНЕЙНОЙ регрессии нужно будет составить уравнение прямой y = a * x + b

X = velocities # список значений угловых скоростей
Y = mean_voltages # средних значений сигнала (напряжения)



# в общем нужно найти такие коээфициенты a и b прямой, который лучше всего описывают зависимость зависимость Y от X
a, b = linear_regression(X, Y)

S = a  # масштабный коэффициент из регрессии
B = b  # смещение из регрессии

# Вычисляем нелинейность по формуле
nl_values, avg_nl, ideal_voltages = calculate_nonlinearity_by_formula(
    rate_values=velocities,     # X - эталонные скорости
    gyro_values=mean_voltages,  # Y - измеренные напряжения
    scale_factor=S,
    bias=B
)


predicted_Y = [a * x + b for x in X] # Предсказанные значения сигнала по найденной прямой - это Y


scale_factor = a * 1000  # потому что y = a * x + b → коэффициент при x это масштабный фактор

zero_index = X.index(0) # - берем значене сигнала для нулевой скорости, чтобы получить смещение нуля
zero_offset_mk = Y[zero_index] * 1000


sum_for_calc_mean = 0
count_if_not_zero = 0
for i in range(len(Y)):
    real_vel = Y[i] * 1000 / scale_factor - zero_offset_mk # реальная скорость с учетом смещения нуля
    if(abs(X[i]) > 0):
        count_if_not_zero += 1
    sum_for_calc_mean += abs(real_vel - X[i]) # здесь у нас есть реальная скорость и скорость из файла - нужно понять раницу в °/с суммируем
mean_sum = sum_for_calc_mean / count_if_not_zero # и считаем среднее

# Разделение на левую и правую части для расчёта асимметрии
# Левая часть: от начала до нуля (включительно)
X_left = X[:zero_index + 1]
Y_left = Y[:zero_index + 1]
# Правая часть: от нуля (включительно) до конца
X_right = X[zero_index:]
Y_right = Y[zero_index:]

# Линейная регрессия для правой части
a_right, b_right = linear_regression(X_right, Y_right)

# Линейная регрессия для левой части
a_left, b_left = linear_regression(X_left, Y_left)

# Асимметрия между наклонами
asymmetry = ((a_left - a_right) / ((a_left + a_right) / 2)) * 100

plt.figure(figsize=(10, 6))
plt.title(f"Масштабный коэффициент: ")
plt.plot(X, Y, 'o', label='Измеренные точки')
plt.plot(X, predicted_Y, '-')
plt.xlabel("Скорость (°/с)")
plt.ylabel("Напряжение (В)")
plt.grid(True)
plt.legend()
plt.tight_layout()

print("Результаты анализа:")
print(f"  Масштабный коэффициент: {scale_factor:.1f} мВ/(°/с)")
print(f"  Нелинейность: {avg_nl:.4f} %")
print(f"  Несимметричность: {asymmetry:.3f} %")
print(f"  Смещение нуля: {zero_offset_mk:.5f} мВ")

plt.figure(figsize=(8, 6))
plt.title(f"Нелинейность масштабного коэффициента: ")
plt.grid(True)
plt.plot(X, nl_values, '-')
plt.tight_layout()

# ============================================
# 2. ОБРАБОТКА ДРЕЙФА (Дрейф.dat)
# ============================================

# Загрузка данных из файла
# выгружаем время и напряжения
time, _, voltage = read_table('Дрейф.dat', time_col=0, voltage_col=1)
print("2. ОБРАБОТКА ДРЕЙФА (Дрейф.dat)")

# Конвертация в угловую скорость (°/с)
# создаём пустой список с угловыми скоростями гироскопа
gyro_rate = []

# для каждого значения напряжения вычисляем угловую скорость
# вычисленную скорость добавляем в список
for v in voltage:
    # осуществляем перевод значение сигнала из В с помощью МК
    # угловая скорость в градусах в секунду
    gyro_rate.append(v / (scale_factor / 1000))

# зная, что в файле измерения проведены для неподвижного гироскопа
# Вычислим смещение нуля (Bias)

# Bias = sum(Wi)/N
zero_offset = sum(gyro_rate) / len(gyro_rate)
print(f"Смещение нуля: {zero_offset * scale_factor:.6f} мВ")

# регрессия для определения тренда дрейфа(по МНК)
# возвращает Trend -- угловой к-т (наклон) линии регресии (град/сек^2)
# intercept = точка пересечения с осью Y
trend, intercept = linear_regression(time, gyro_rate)

# Переводим тренд из (град/сек^2) в (°/ч/ч)
# Умножаем на квадрат количества секунд в часе (3600²)
# Это стандартный формат представления дрейфа гироскопов
trend_per_hour2 = trend * 3600 ** 2
# !!! Вывод значения тренда !!!
print(f"Тренд дрейфа по всему сигналу: {trend_per_hour2:.4f} °/ч/ч")

# расчет тренда от начала до n - минут START
minute = 2
begin_window_time, begin_window_trend = calc_trend_from_begin_to_minute(time, gyro_rate, minute)
print("\nРассчет тренда от начала до ", minute, " минут")
print(f"Тренд дрейфа: {begin_window_trend:.4f} °/ч/ч")
# расчет тренда от начала до n - минут END

#тренд по временным интервалам START
mean_window_time, mean_window_trend = mean_in_n_min_windows(time, gyro_rate, 1)
plt.figure(figsize=(10, 6))
plt.plot(mean_window_time, mean_window_trend)
plt.xlabel("Время (с)")
plt.ylabel("Тренд (град/ч²)")
plt.title("Динамика тренда гироскопа вычисление среднего значения на интервале")
plt.tight_layout()
plt.grid(True)

print("\nРассчет тренда по временным интервалам ")
print(f"Тренд дрейфа минимум: {min(mean_window_trend):.4f} °/ч/ч")
print(f"Тренд дрейфа максимум: {max(mean_window_trend):.4f} °/ч/ч")
print(f"Тренд дрейфа среднее: {sum(mean_window_trend) / len(mean_window_trend):.4f} °/ч/ч")
#тренд по временным интервалам END

# Расчёт девиации Аллана для анализа шумов
# вычисляем период дискретизации (разница между временными метками)
ts = (time[1] - time[0])

# Частота дискретизации (сколько измерений в секунду)
fs = 1 / ts

# вызов рукопашной девиации Аллана
taus, adevs = allan_deviation(gyro_rate, fs)

# ДЛЯ ПРОВЕРКИ расчёт девиации Аллана через библиотеку allantools
# возвращаемые значения
# taus_tool - массив интервалов усреднения τ
# adevs_tool - значение девиации Аллана σ(τ) для каждого τ
# errors - погрешности расчёта для каждого σ(τ).
# counts - количество точек, использованных для расчёта каждого σ(τ)
(taus_tool, adevs_tool, errors, counts) = at.adev(
    np.array(gyro_rate), # конвертируем список gyro_rate в массив numpy
    rate=fs, # задаем частоту дискретизации
    data_type="freq", # входные данные -- частотные измерения(угловые скорости гироскопа)
    taus=taus  # автоматический выбор tau -- массив интервалов усреднения для которыз считается девиация
)

# Логарифмирование данных для линеаризации графиков ARW RRW - это все линии, поэтому и логарифмируем
# Логарифирование нужно, чтобы выделить линейные участки графика
log_taus = [math.log10(t) for t in taus]
log_adevs = [math.log10(a) for a in adevs]

# Выделение участков шумов
# Примерные участки
# ARW (Angular Random Walk) - начальный участок (первые 33% точек) - короткие τ
first_33_percent = int(0.33 * len(taus))
idx_arw = [i for i, t in enumerate(taus) if t < 10 and i < first_33_percent]

# BI участок (плато, обычно 10-1000 сек)
idx_bi = [i for i, t in enumerate(taus) if 10 <= t <= 1000]

# RRW (Rate Random Walk) - последние 10% точек (длинные τ))
last_10_percent = int(0.9 * len(taus))
idx_rrw = [i for i, t in enumerate(taus) if t > 1000 and i > last_10_percent]

# ARW доминирует на малых τ, RRW — на больших.

# Линейная регрессия для ARW (наклон -0.5 в логарифмическом масштабе)
x_arw = [log_taus[i] for i in idx_arw]
y_arw = [log_adevs[i] for i in idx_arw]
a_arw, b_arw = linear_regression(x_arw, y_arw)

# Расчет ARW в °/√ч
# формула взята из IEEE 952-1997, IEEE Std 647-2006 умножение на * math.sqrt(3600) перевод в °/√ч
arw_base = 10 ** b_arw
arw_hours = arw_base * math.sqrt(3600)  # °/√ч

# Bias Instability - минимальное значение девиации (плато на графике)
# Bias Instability — это минимальный уровень шума,
# вызванный фликкер-шумом (дрейфом электроники).
# Единицы: °/с.
bi_values = [adevs[i] for i in idx_bi]
bi_base = min(bi_values)  # °/с
bi_hours = bi_base * 3600  # °/ч

# Линейная регрессия для RRW (наклон +0.5 в логарифмическом масштабе)
x_rrw = [log_taus[i] for i in idx_rrw]
y_rrw = [log_adevs[i] for i in idx_rrw]
a_rrw, b_rrw = linear_regression(x_rrw, y_rrw)

# Расчет RRW в °/h³/²
rrw_base = 10 ** b_rrw  # °/с^(3/2)
rrw_hours = rrw_base * math.sqrt(3) * (3600 ** 1.5)  # °/ч^(3/2)

# вывод результатов для ARW, BI, RRW
print("\nЗначения, полученные из allan_deviation -- ручного расчёта")
print(f"Angular Random Walk (ARW): {arw_hours:.6f} °/√ч")
print(f"Bias Instability (BI): {bi_hours:.6f} °/ч")
print(f"Rate Random Walk (RRW): {rrw_hours:.6f} °/ч^(3/2)")

# РАСЧЕТ ПОЛИНОМА ДИСПЕРСИИ АЛЛАНА START
# степени полинома дисперсии Аллана
powers = [-2, -1, 0, 1]
#Функция вычисляет коэффициенты Q, N, B, K по всему участку (на интервале t_min < τ < t_max).
#Она использует least squares (метод наименьших квадратов) для нахождения этих коэффициентов.
coeffs, model, taus_filtered = plot_allan_approximation(taus, adevs, powers)

Q_coeff, N_coeff, B_coeff, K_coeff = coeffs[:4]

# Формулы согласно IEEE
ARW_correct = math.sqrt(abs(N_coeff)) * math.sqrt(3600)  # °/√ч
BI_correct = math.sqrt(abs(B_coeff)) * 3600  # °/ч
RRW_correct = math.sqrt(abs(K_coeff)) / math.sqrt(3) * (3600 ** 1.5)  # °/ч^(3/2)

# print("\nк-ты подсчитанные при помощи plot_allan_approximation -- полинома")
# print(f"Angular Random Walk: {ARW_correct:.6f} °/√ч")
# print(f"Bias Instability: {BI_correct:.6f} °/ч")
# print(f"Rate Random Walk: {RRW_correct:.6f} °/ч^(3/2)")
# РАСЧЕТ ПОЛИНОМА ДИСПЕРСИИ АЛЛАНА END


# График сырых данных (дрейф)
plt.figure(figsize=(12, 6))
plt.subplot(121)
gyro_rate_zeroed = [(g - zero_offset) for g in gyro_rate]
plt.plot(time, gyro_rate_zeroed)
plt.title('Измерения дрейфа')
plt.xlabel('Время (с)')
plt.ylabel('Угловая скорость (°/с)')
plt.grid(True)

# График Девиации Аллана с аппроксимацией
plt.subplot(122)

# Переводим все значения в °/ч (умножаем на 3600)
plt.loglog(taus, [a * 3600 for a in adevs], 'bo-', label='Данные (ручной расчёт)')
plt.loglog(taus_tool, [a * 3600 for a in adevs_tool], 'r*-', label='allantools')
plt.loglog(taus_filtered, [math.sqrt(m) * 3600 for m in model], '--', color="red", label="Модель (сумма шумов)")

#plt.loglog(taus, model_adevs, 'k--', linewidth=2, label='Модель Аллана')
# plt.loglog(taus, [10**(a_arw*math.log10(t) + b_arw) for t in taus], 'r--', label='ARW')
# plt.loglog(taus, [bi]*len(taus), 'g--', label='Bias Instability')
# plt.loglog(taus, [10**(a_rrw*math.log10(t) + b_rrw) for t in taus], 'm--', label='RRW')
plt.title('Девиация Аллана с аппроксимацией')
plt.xlabel('Время усреднения τ (с)')
plt.ylabel('Девиация Аллана (°/ч)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()
