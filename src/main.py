import argparse
import matplotlib.pyplot as plt
import numpy as np
import math

from reader import read_first_three_columns
from drift_reader import read_drift_file
from metrics import calc_scale_metrics, calc_drift_metrics, polyfit_allan, plot_allan_approximation

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", help="Path to scale data file")
    parser.add_argument("--drift", help="Path to drift data file")
    parser.add_argument("--plot", action="store_true", help="Show plots")
    args = parser.parse_args()

    SF = None
    if args.scale:
        analyze_scale(args.scale, do_plot=args.plot)
        # Получаем масштабный коэффициент из анализа
        # Здесь упрощённо: читаем файл и пересчитываем SF
        _, omega, U = read_first_three_columns(args.scale)
        M = calc_scale_metrics(omega, U)
        SF = M['SF']

    if args.drift:
        if SF is None:
            print("Warning: scale factor not provided, using SF=1.0 mV/(°/s)")
            SF = 1.0
        analyze_drift(args.drift, SF, do_plot=args.plot)

    if args.plot:
        input("Press Enter to exit...")

