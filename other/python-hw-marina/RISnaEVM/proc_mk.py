import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Загрузка данных из файла
df = pd.read_excel(r'Data\MK.xlsx', sheet_name='Sheet1')


# 1. Расчет масштабного коэффициента (МК)
def calculate_scale_factor(df):
    """
    Расчет масштабного коэффициента гироскопа.
    МК = (максимальное напряжение - минимальное напряжение) / (2 * угловая скорость)
    Так как реальная угловая скорость равна нулю, используем известное значение rate (-50°/с)
    """
    # Извлекаем данные гироскопов
    gyr_x = df['GYR_X_865'].values
    gyr_y = df['GYR_Y_871'].values
    gyr_z = df['GYR_Z_925'].values

    # Известная угловая скорость (в градусах/сек)
    omega = -50  # °/с

    # Рассчитываем МК для каждого канала
    mk_x = (np.max(gyr_x) - np.min(gyr_x)) / (2 * omega) * 1000  # мВ/(°/с)
    mk_y = (np.max(gyr_y) - np.min(gyr_y)) / (2 * omega) * 1000  # мВ/(°/с)
    mk_z = (np.max(gyr_z) - np.min(gyr_z)) / (2 * omega) * 1000  # мВ/(°/с)

    return mk_x, mk_y, mk_z, gyr_x, gyr_y, gyr_z


# 2. Расчет нелинейности
def calculate_nonlinearity(gyr_data, mk, omega):
    """
    Расчет нелинейности масштабного коэффициента.
    Нелинейность = (макс отклонение от линейной зависимости) / (размах выходного сигнала) * 100%
    """
    # Линейная модель
    linear_model = lambda x, a, b: a * x + b

    # Создаем массив угловых скоростей (все значения равны omega)
    omega_array = np.full_like(gyr_data, omega)

    # Аппроксимируем данные линейной моделью
    popt, _ = curve_fit(linear_model, omega_array, gyr_data)

    # Рассчитываем отклонения от линейной модели
    deviations = gyr_data - linear_model(omega_array, *popt)

    # Максимальное отклонение
    max_deviation = np.max(np.abs(deviations))

    # Размах выходного сигнала
    signal_range = np.max(gyr_data) - np.min(gyr_data)

    # Нелинейность в процентах
    nonlinearity = (max_deviation / signal_range) * 100

    return nonlinearity, deviations


# 3. Расчет несимметричности
def calculate_asymmetry(gyr_data, mk, omega):
    """
    Расчет несимметричности масштабного коэффициента.
    Несимметричность = |МК_положительный - МК_отрицательный| / средний_МК * 100%
    В нашем случае все измерения при одной угловой скорости,
    поэтому используем отклонения от среднего.
    """
    # Среднее значение выходного сигнала
    mean_value = np.mean(gyr_data)

    # Положительные и отрицательные отклонения
    positive_dev = np.mean(gyr_data[gyr_data > mean_value] - mean_value)
    negative_dev = np.mean(mean_value - gyr_data[gyr_data < mean_value])

    # Рассчитываем несимметричность
    asymmetry = np.abs(positive_dev - negative_dev) / ((positive_dev + negative_dev) / 2) * 100

    return asymmetry


# 4. Расчет смещения нуля
def calculate_zero_offset(gyr_data, mk):
    """
    Расчет смещения нуля гироскопа.
    Смещение нуля = среднее значение выходного сигнала / масштабный коэффициент
    """
    zero_offset = np.mean(gyr_data) / (mk / 1000)  # переводим мВ в В
    return zero_offset


# Основные расчеты
mk_x, mk_y, mk_z, gyr_x, gyr_y, gyr_z = calculate_scale_factor(df)
omega = -50  # известная угловая скорость в °/с

# Расчет нелинейности
nl_x, dev_x = calculate_nonlinearity(gyr_x, mk_x, omega)
nl_y, dev_y = calculate_nonlinearity(gyr_y, mk_y, omega)
nl_z, dev_z = calculate_nonlinearity(gyr_z, mk_z, omega)

# Расчет несимметричности
asym_x = calculate_asymmetry(gyr_x, mk_x, omega)
asym_y = calculate_asymmetry(gyr_y, mk_y, omega)
asym_z = calculate_asymmetry(gyr_z, mk_z, omega)

# Расчет смещения нуля
zero_x = calculate_zero_offset(gyr_x, mk_x)
zero_y = calculate_zero_offset(gyr_y, mk_y)
zero_z = calculate_zero_offset(gyr_z, mk_z)

# Вывод результатов
print("Результаты расчетов:")
print(f"Масштабный коэффициент X: {mk_x:.2f} мВ/(°/с)")
print(f"Масштабный коэффициент Y: {mk_y:.2f} мВ/(°/с)")
print(f"Масштабный коэффициент Z: {mk_z:.2f} мВ/(°/с)")
print()
print(f"Нелинейность X: {nl_x:.2f}%")
print(f"Нелинейность Y: {nl_y:.2f}%")
print(f"Нелинейность Z: {nl_z:.2f}%")
print()
print(f"Несимметричность X: {asym_x:.2f}%")
print(f"Несимметричность Y: {asym_y:.2f}%")
print(f"Несимметричность Z: {asym_z:.2f}%")
print()
print(f"Смещение нуля X: {zero_x:.2f} °/с")
print(f"Смещение нуля Y: {zero_y:.2f} °/с")
print(f"Смещение нуля Z: {zero_z:.2f} °/с")

# Построение графиков
time = df['time'].values

# Графики измерений
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(time, gyr_x, label='GYR_X')
plt.title('Измерения гироскопа по оси X')
plt.xlabel('Время, с')
plt.ylabel('Напряжение, В')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, gyr_y, label='GYR_Y')
plt.title('Измерения гироскопа по оси Y')
plt.xlabel('Время, с')
plt.ylabel('Напряжение, В')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, gyr_z, label='GYR_Z')
plt.title('Измерения гироскопа по оси Z')
plt.xlabel('Время, с')
plt.ylabel('Напряжение, В')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('Plots\Сырые_измерения_МК.png')
plt.show()

# Графики нелинейности
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(time, dev_x, label='Отклонения X')
plt.title('Нелинейность по оси X')
plt.xlabel('Время, с')
plt.ylabel('Отклонение от линейности, В')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, dev_y, label='Отклонения Y')
plt.title('Нелинейность по оси Y')
plt.xlabel('Время, с')
plt.ylabel('Отклонение от линейности, В')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, dev_z, label='Отклонения Z')
plt.title('Нелинейность по оси Z')
plt.xlabel('Время, с')
plt.ylabel('Отклонение от линейности, В')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('Plots\Графики_нелинейности_МК.png')
plt.show()
