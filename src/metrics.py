# metrics.py
import numpy as np
import allantools as at

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
    zero_offset = b / a_raw

    return {
        'SF': SF,
        'Linearity_%': nonlin,
        'Asymmetry_%SF': asym,
        'ZeroOffset_°/s': zero_offset,
        'a_raw': a_raw,
        'b': b
    }

''' 1
def calc_drift_metrics(times, rates):
    """
    Вычисляет параметры дрейфа:
      Zero Offset       — среднее (IEEE 528 §2.122),
      Trend °/h²        — коэффициент лин. тренда·3600²,
      Allan σ(τ)        — см. IEEE 952 Annex C (C.1),
      ARW, BI, RRW      — см. IEEE 952 Table B.1.

    Алгоритм Allan σ:
      τ0 = Δt
      m = τ/τ0
      yk = avg of rates[k·m:(k+1)·m]
      σ²(τ) = ½·mean((yk+1−yk)²)
    """
    # 1) нулевое смещение °/с
    Z = rates.mean()

    # 2) тренд по лин. регрессии rates ≈ a·t + b -> °/h²
    a, _ = np.polyfit(times, rates, 1)
    trend = a * (3600**2)

    # 3) Allan deviation
    τ0 = times[1] - times[0]
    N = len(rates)
    max_m = N // 2
    m_vals = np.unique(np.logspace(0, np.log10(max_m), num=20, dtype=int))
    taus, sigmas = [], []
    for m in m_vals:
        M = N // m
        if M < 2: continue
        y = rates[:M*m].reshape(M, m).mean(axis=1)
        σ2 = 0.5 * np.mean(np.diff(y)**2)
        taus.append(m * τ0)
        sigmas.append(np.sqrt(σ2))
    taus = np.array(taus)
    sigmas = np.array(sigmas)

    # 4) ARW (slope≈−½ on log–log)
    idx_arw = slice(0, 5)
    p_arw = np.polyfit(np.log10(taus[idx_arw]), np.log10(sigmas[idx_arw]), 1)
    ARW = 10**p_arw[1]

    # 5) Bias Instability = min σ(τ)
    BI = sigmas.min()

    # 6) RRW (slope≈+½ on log–log)
    idx_rrw = slice(-5, None)
    p_rrw = np.polyfit(np.log10(taus[idx_rrw]), np.log10(sigmas[idx_rrw]), 1)
    RRW = 10**p_rrw[1]

    return {
        'ZeroOffset_°/s': Z,
        'Trend_°/h²': trend,
        'taus': taus,
        'sigmas': sigmas,
        'ARW': ARW,
        'BI': BI,
        'RRW': RRW
    }
'''

import numpy as np
import allantools as at

'''
def calc_drift_metrics(times, rates, use_allantools=False):
    """
    Вычисляет параметры дрейфа двумя методами:
      1) вручную (Allan σ ручным методом),
      2) через библиотеку allantools.

    Возвращается словарь:
    {
       'manual': {
           'ZeroOffset_°/s': Z_manual,
           'Trend_°/h²': trend_manual,
           'taus': taus_manual,
           'sigmas': sigmas_manual,
           'ARW': ARW_manual,
           'BI': BI_manual,
           'RRW': RRW_manual
       },
       'allantools': {
           'taus': taus_tool,
           'sigmas': sigmas_tool,
           'ARW': ARW_tool,
           'BI': BI_tool,
           'RRW': RRW_tool
       }
    }
    """
    times = np.asarray(times)
    rates = np.asarray(rates)

    # ---- Ручной метод ----
    # 1) Zero Offset
    Z_manual = rates.mean()
    # 2) Trend в °/h²
    a, _ = np.polyfit(times, rates, 1)
    trend_manual = a * (3600**2)
    # 3) Allan deviation ручным методом
    τ0 = times[1] - times[0]
    N = rates.size
    max_m = N // 2
    m_vals = np.unique(np.logspace(0, np.log10(max_m), num=50, dtype=int))
    taus_manual, sigmas_manual = [], []
    for m in m_vals:
        M = N // m
        if M < 2:
            continue
        y = rates[:M*m].reshape(M, m).mean(axis=1)
        delta = np.diff(y)
        sigma2 = 0.5 * np.mean(delta * delta)
        taus_manual.append(m * τ0)
        sigmas_manual.append(np.sqrt(sigma2))
    taus_manual = np.array(taus_manual)
    sigmas_manual = np.array(sigmas_manual)
    # 4) ARW вручную
    idx_arw = slice(0, max(3, len(taus_manual)//10))
    p_arw = np.polyfit(np.log10(taus_manual[idx_arw]), np.log10(sigmas_manual[idx_arw]), 1)
    ARW_manual = 10**p_arw[1] * np.sqrt(3600)
    # 5) BI вручную
    BI_manual = sigmas_manual.min() * 3600
    # 6) RRW вручную
    idx_rrw = slice(-max(3, len(taus_manual)//10), None)
    p_rrw = np.polyfit(np.log10(taus_manual[idx_rrw]), np.log10(sigmas_manual[idx_rrw]), 1)
    RRW_manual = 10**p_rrw[1] * np.sqrt(3) * (3600**1.5)

    # ---- Метод через allantools ----
    fs = 1.0 / τ0
    taus_tool, adevs_tool, _, _ = at.adev(data=rates, rate=fs, data_type='freq', taus=taus_manual)
    # ARW через библиотеку
    idx_arw_t = taus_tool < 10
    p_arw_t = np.polyfit(np.log10(taus_tool[idx_arw_t]), np.log10(adevs_tool[idx_arw_t]), 1)
    ARW_tool = 10**p_arw_t[1] * np.sqrt(3600)
    # BI через библиотеку
    BI_tool = adevs_tool.min() * 3600
    # RRW через библиотеку
    idx_rrw_t = taus_tool > 1000
    p_rrw_t = np.polyfit(np.log10(taus_tool[idx_rrw_t]), np.log10(adevs_tool[idx_rrw_t]), 1)
    RRW_tool = 10**p_rrw_t[1] * np.sqrt(3) * (3600**1.5)

    result = {
        'manual': {
            'ZeroOffset_°/s': Z_manual,
            'Trend_°/h²': trend_manual,
            'taus': taus_manual,
            'sigmas': sigmas_manual,
            'ARW': ARW_manual,
            'BI': BI_manual,
            'RRW': RRW_manual
        },
        'allantools': {
            'taus': taus_tool,
            'sigmas': adevs_tool,
            'ARW': ARW_tool,
            'BI': BI_tool,
            'RRW': RRW_tool
        }
    }

    return {
        'ZeroOffset_°/s': Z_manual,
        'Trend_°/h²': trend_manual,
        'taus': taus_manual,
        'sigmas': sigmas_manual,
        'ARW': ARW_tool,
        'BI': BI_tool,
        'RRW': RRW_tool
    }
'''

import numpy as np
import math

def allan_deviation(data, fs):
    """
    Вычисляет стандартную девиацию Аллана для ряда data и частоты дискретизации fs.
    """
    n = len(data)
    max_m = n // 2

    m_vals = []
    m = 1
    while m <= max_m:
        m_vals.append(m)
        m = int(math.ceil(m * 1.2))

    taus = []
    adevs = []
    for m in m_vals:
        num_blocks = n // m
        if num_blocks < 2:
            break

        # Уздываем средние по неперекрывающимся блокам длины m
        means = [np.mean(data[i*m:(i+1)*m]) for i in range(num_blocks)]
        diffs = np.diff(means)
        sigma2 = 0.5 * np.mean(diffs**2)
        taus.append(m / fs)
        adevs.append(math.sqrt(sigma2))

    return np.array(taus), np.array(adevs)

def calc_drift_metrics(times, rates):
    """
    Вычисляет параметры дрейфа гироскопа:
      - Zero Offset      [°/s]
      - Trend            [°/h²]
      - Allan Deviation  (τ, σ(τ))
      - Angular Random Walk (ARW) [°/√h]
      - Bias Instability   (BI) [°/h]
      - Rate Random Walk   (RRW) [°/h³/²]
    """
    # 1) Zero Offset
    Z = np.mean(rates)

    # 2) Trend в °/h²
    # polyfit вернёт [a, b] для rates ≈ a*t + b
    a, b = np.polyfit(times, rates, 1)
    trend = a * (3600**2)

    # 3) Allan Deviation
    fs = 1.0 / (times[1] - times[0])
    taus, sigmas = allan_deviation(rates, fs)

    log_taus = np.log10(taus)
    log_sig  = np.log10(sigmas)
    N = len(taus)

    # 4) ARW: первые 33% точек
    n_arw = max(2, int(0.33 * N))
    coeffs_arw = np.polyfit(log_taus[:n_arw], log_sig[:n_arw], 1)
    b_arw = coeffs_arw[1]
    ARW = 10**b_arw * math.sqrt(3600)

    # 5) BI: минимум σ(τ) → °/h
    BI = sigmas.min() * 3600

    # 6) RRW: последние 10% точек
    start_rrw = max(0, N - max(2, int(0.10 * N)))
    coeffs_rrw = np.polyfit(log_taus[start_rrw:], log_sig[start_rrw:], 1)
    b_rrw = coeffs_rrw[1]
    RRW = 10**b_rrw * math.sqrt(3) * (3600**1.5)

    return {
        'ZeroOffset_°/s': Z,
        'Trend_°/h²': trend,
        'taus': taus,
        'sigmas': sigmas,
        'ARW': ARW,
        'BI': BI,
        'RRW': RRW
    }

