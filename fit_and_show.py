import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-(x - mean)**2 / (2 * stddev**2))

def fit(noise, factor=1):
    bins_noise = noise.max() - noise.min()
    bins, bin_edges = np.histogram(noise, bins=int(bins_noise / factor))
    # print(bins)

    initial_amplitude = np.max(bins)
    initial_std = np.std(noise)
    initial_mean = np.mean(noise)

    initial_guess = [initial_amplitude, initial_mean,
                     initial_std]  # Initial values for amplitude, mean, and standard deviation
    params, covariance, infodict, _, _ = curve_fit(gaussian, bin_edges[:-1], bins, p0=initial_guess, full_output=True)

    # Extract the fitted parameters
    amplitude_fit, mean_fit, stddev_fit = params

    residuals = infodict['fvec']

    return bins, bin_edges, amplitude_fit, mean_fit, stddev_fit, residuals

if __name__ == '__main__':
    df700 = pd.read_csv('./Day 2a Trigger threshold_Apogee/export_MQ_700V_LALS_triggerFITC_0.csv')
    noise = df700['Large Angle Light Scatter']
    noise = noise[noise > 1]
    noise = noise[noise < 100000]

    bins, bin_edges, amplitude_fit, mean_fit, stddev_fit, residuals = fit(noise, factor=10)
    y_fit = gaussian(bin_edges[:-1], amplitude_fit, mean_fit, stddev_fit)

    fig, axs = plt.subplots(1, 1, figsize=(10, 10))

    axs.step(bin_edges[:-1], bins, linewidth=4, label='Data')
    axs.step(bin_edges[:-1], y_fit, linewidth=4, label='Fit')

    plt.xlim(1, 100000)
    plt.xlabel('PE (a.u.)')
    plt.ylabel('Counts')
    plt.show()
    print(np.sum(residuals ** 2))
    lod = mean_fit + (4 * stddev_fit)
    lod / (2 ** 6)