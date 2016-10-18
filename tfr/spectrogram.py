import numpy as np
import math
import os.path
import scipy

from .files import load_wav
from .features import mean_power
from .analysis import split_to_blocks

def magnitude_spectrum(x):
    '''
    Magnitude spectrum scaled so that each bin corresponds to the original sine
    amplitude. Only the real part of the spectrum is returned.
    x - 1D sampled signal (possibly already windowed)
    '''
    X = np.fft.fft(x)
    Xr = real_half(X)
    N = Xr.shape[-1]
    return abs(Xr) / N

def real_half(X):
    N = X.shape[1]
    return np.hstack([0.5 *  X[:, :1], X[:, 1:N//2]])

def energy_weighted_spectrum(x):
    N = x.shape[-1]
    X = np.fft.fft(x)
    # np.allclose(energy(abs(X) / math.sqrt(N)), energy(x))
    # np.allclose(energy(abs(X[:N//2]) / math.sqrt(N//2)), energy(x))
    return abs(X) / math.sqrt(N)

def normalize_mean_power(x):
    '''
    Normalize a vector so that it has energy equal to its length,
    ie. mean power equal to 1.0.
    Useful to normalize the FFT window.

    np.allclose(normalize_window(x), len(x))
    '''
    return x / mean_power(x)

def create_window(size):
    w = scipy.hanning(size)
    w = w / mean_power(w)
    return w

def spectrogram(filename, block_size=2048, hop_size=512, to_log=True):
    song, fs = load_wav(filename)
    x, times = split_to_blocks(song, block_size, hop_size=hop_size)
    w = create_window(block_size)
    X = magnitude_spectrum(x * w) ** 2
    if to_log:
        # dbFB
        X = 20 * np.log10(np.maximum(1e-6, X))
    return X, x, times

def fftfreqs(block_size, fs):
    return np.fft.fftfreq(block_size, 1/fs)[:block_size // 2]

def inverse_spectrum(spectrum, window):
    '''
    inverse_spectrum(np.fft.fft(x * window), window) == x
    '''
    return np.real(np.fft.ifft(spectrum)) / window
