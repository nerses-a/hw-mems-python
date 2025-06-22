# metrics.py
import numpy as np



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
    nonlin = 100.0 * deviations.max() / U_FS

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

