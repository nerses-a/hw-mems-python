# import dreif
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. Загрузка и подготовка данных
data = pd.read_excel('Data\Дрейф.xlsx', sheet_name='Sheet1')

# Выбираем только гироскопические данные (столбцы с GYR)
gyro_cols = [col for col in data.columns if 'GYR' in col]
gyro_data = data[['time'] + gyro_cols]

# Переименуем столбцы для удобства
gyro_data.columns = ['time', 'GYR_X', 'GYR_Y', 'GYR_Z']

# 2. График "сырых" измерений (°/с)
plt.figure(figsize=(12, 6))
for axis in ['GYR_X', 'GYR_Y', 'GYR_Z']:
    plt.plot(gyro_data['time'], gyro_data[axis], label=axis)
plt.xlabel('Время, с')
plt.ylabel('Угловая скорость, °/с')
plt.title('Сырые измерения гироскопа')
plt.grid(True)
plt.legend()
plt.savefig('Plots\Сырые_измерения_дрейф.png')
plt.show()

# 3. Расчет смещения нуля (bias)
bias = gyro_data[['GYR_X', 'GYR_Y', 'GYR_Z']].mean()
print("\nСмещение нуля (°/с):")
print(bias)


# 4. Расчет тренда (линейный дрейф)
def linear_trend(x, a, b):
    return a * x + b


trend_params = {}
for axis in ['GYR_X', 'GYR_Y', 'GYR_Z']:
    params, _ = curve_fit(linear_trend, gyro_data['time'], gyro_data[axis])
    trend_params[axis] = params
    # Тренд в °/с^2 (первая производная)
    trend_rate = params[0]  # °/с^2
    # Переводим в °/ч/ч: умножаем на 3600*3600
    trend_rate_hrhr = trend_rate * 3600 * 3600
    print(f"\nТренд для {axis}: {trend_rate_hrhr:.6f} °/ч/ч")


# 5. Расчет девиации Аллана
def allan_deviation(omega, fs, tau):
    """
    Расчет девиации Аллана
    :param omega: массив угловых скоростей (°/с)
    :param fs: частота дискретизации (Гц)
    :param tau: массив временных интервалов для расчета (с)
    :return: массив значений девиации Аллана для каждого tau
    """
    n = len(omega)
    t = np.arange(n) / fs
    adev = np.zeros_like(tau, dtype=float)

    for i, m in enumerate(tau):
        m = int(m * fs)  # Количество точек в интервале m
        if m == 0:
            m = 1
        d = n // m  # Количество групп

        # Разбиваем данные на группы по m точек
        groups = omega[:d * m].reshape(d, m)

        # Вычисляем средние для каждой группы
        group_means = groups.mean(axis=1)

        # Разности между соседними средними
        diffs = np.diff(group_means)

        # Девиация Аллана
        adev[i] = np.sqrt(0.5 * np.mean(diffs ** 2))

    return adev


# Частота дискретизации (из данных - 100 Гц)
fs = 1 / (gyro_data['time'][1] - gyro_data['time'][0])

# Временные интервалы для расчета (логарифмическая шкала)
tau = np.logspace(-2, np.log10(len(gyro_data) / fs / 2), 100)

# Расчет для каждой оси
adev_results = {}
for axis in ['GYR_X', 'GYR_Y', 'GYR_Z']:
    adev = allan_deviation(gyro_data[axis].values, fs, tau)
    adev_results[axis] = adev

# 6. График девиации Аллана
plt.figure(figsize=(12, 6))
for axis in ['GYR_X', 'GYR_Y', 'GYR_Z']:
    plt.loglog(tau, adev_results[axis], label=axis)
plt.xlabel('Временной интервал, τ (с)')
plt.ylabel('Девиация Аллана, σ(τ) (°/с)')
plt.title('Девиация Аллана')
plt.grid(True, which="both", ls="-")
plt.legend()
plt.savefig('Plots\Девиация_Аллана_дрейф.png')
plt.show()


# 7. Аппроксимация девиации Аллана
def allan_variance_model(tau, N, B, K):
    """
    Модель для аппроксимации девиации Аллана
    :param tau: временной интервал
    :param N: коэффициент Angular Random Walk
    :param B: коэффициент Bias Instability
    :param K: коэффициент Rate Random Walk
    :return: значение девиации Аллана
    """
    return np.sqrt(N ** 2 / tau + B ** 2 + K ** 2 * tau / 3)


# Аппроксимация для каждой оси
params_results = {}
plt.figure(figsize=(12, 6))

for axis in ['GYR_X', 'GYR_Y', 'GYR_Z']:
    # Начальные приближения для параметров
    p0 = [1e-3, 1e-3, 1e-3]

    # Аппроксимация
    try:
        params, _ = curve_fit(allan_variance_model, tau, adev_results[axis], p0=p0)
        params_results[axis] = params

        # Расчет аппроксимированной кривой
        adev_fit = allan_variance_model(tau, *params)

        # График
        plt.loglog(tau, adev_results[axis], 'o', markersize=4, label=f'{axis} (данные)')
        plt.loglog(tau, adev_fit, label=f'{axis} (аппроксимация)')

        # Вывод параметров
        print(f"\nПараметры для {axis}:")
        print(f"Angular Random Walk (N): {params[0]:.6f} °/√s")
        print(f"Bias Instability (B): {params[1]:.6f} °/s")
        print(f"Rate Random Walk (K): {params[2]:.6f} °/s^{3 / 2}")

    except Exception as e:
        print(f"Ошибка при аппроксимации {axis}: {str(e)}")

plt.xlabel('Временной интервал, τ (с)')
plt.ylabel('Девиация Аллана, σ(τ) (°/с)')
plt.title('Девиация Аллана с аппроксимацией')
plt.grid(True, which="both", ls="-")
plt.legend()
plt.savefig('Plots\Девиация_Аллана_с_аппроксимацией_дрейф.png')
plt.show()
