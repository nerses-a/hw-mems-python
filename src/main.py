# main.py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from reader import read_first_three_columns

def calc_metrics(omega, U):
    U_mV = U * 1e3
    a, b = np.polyfit(omega, U_mV, 1)
    U_FS = a * np.max(np.abs(omega))
    nonlin = 100 * np.max(np.abs(U_mV - (a*omega + b))) / U_FS
    mask_p = omega > 0
    mask_n = omega < 0
    a_p, _ = np.polyfit(omega[mask_p], U_mV[mask_p], 1)
    a_n, _ = np.polyfit(omega[mask_n], U_mV[mask_n], 1)
    asym = 100 * (a_p - a_n) / ((a_p + a_n) / 2)
    zero_offset = b / a
    return {
        'SF': a,
        'Linearity_%FS': nonlin,
        'Asymmetry_%SF': asym,
        'ZeroOffset_°/s': zero_offset,
        'a': a,
        'b': b
    }

def print_results(metrics):
    print(f"Scale Factor:    {metrics['SF']:.3f} mV/(°/s)")
    print(f"Linearity Error: {metrics['Linearity_%FS']:.3f} %FS")
    print(f"Asymmetry:       {metrics['Asymmetry_%SF']:.3f} %SF")
    print(f"Zero Offset:     {metrics['ZeroOffset_°/s']:.3f} °/s")

def analyze_file(filename, do_plot=False):
    t, omega, U = read_first_three_columns(filename)
    metrics = calc_metrics(omega, U)
    if do_plot:
        U_mV = U * 1e3
        a, b = metrics['a'], metrics['b']
        omega_lin = np.linspace(omega.min(), omega.max(), 200)
        plt.figure(); plt.scatter(omega, U_mV, s=10)
        plt.plot(omega_lin, a*omega_lin + b, 'r-')
        plt.xlabel('Ω (°/s)'); plt.ylabel('U (мВ)')
        plt.title('U vs Ω')
        plt.figure(); plt.scatter(omega, U_mV - (a*omega + b), s=10)
        plt.xlabel('Ω (°/s)'); plt.ylabel('Resid (мВ)'); plt.title('Nonlinearity')
        plt.show()
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    results = analyze_file(args.input_file, do_plot=args.plot)
    print_results(results)
