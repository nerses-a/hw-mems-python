# main.py
import argparse
import matplotlib.pyplot as plt
import numpy as np

from reader import read_first_three_columns
from drift_reader import read_drift_file
from metrics import calc_scale_metrics, calc_drift_metrics

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


def analyze_drift(file_drift, do_plot=False):
    t, rates = read_drift_file(file_drift)
    M = calc_drift_metrics(t, rates)
    print("\n=== Drift Metrics ===")
    print(f"Zero Offset:     {M['ZeroOffset_°/s']:.6f} °/s")
    print(f"Trend:           {M['Trend_°/h²']:.6f} °/h²")
    print(f"Angular RW:      {M['ARW']:.6e} °/√h")
    print(f"Bias Instability:{M['BI']:.6e} °/h")
    print(f"Rate RW:         {M['RRW']:.6e} °/h/√h")
    if do_plot:
        plt.figure(); plt.plot(t, rates); plt.title("Raw Drift"); plt.show()
        plt.figure(); plt.loglog(M['taus'], M['sigmas'], 'o-')
        plt.title("Allan Deviation"); plt.show(block=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale")
    parser.add_argument("--drift")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.scale:
        analyze_scale(args.scale, do_plot=args.plot)
    if args.drift:
        analyze_drift(args.drift, do_plot=args.plot)

    input()


