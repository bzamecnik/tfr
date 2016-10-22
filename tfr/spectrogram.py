import numpy as np
import math
import scipy
import soundfile as sf

from .features import mean_power
from .analysis import split_to_blocks, to_mono

def spectrogram(filename, block_size=2048, hop_size=512, to_log=True):
    """
    Computes an STFT magnitude power spectrogram from an audio file.
    Returns: spectrogram, audio_samples, block_times
    """
    song, fs = sf.read(filename)
    song_mono = to_mono(song)
    x, times = split_to_blocks(song_mono, block_size, hop_size=hop_size)
    w = create_window(block_size)
    X = stft_spectrogram(x, w, to_log)
    return X, x, times

def stft_spectrogram(x, w, to_log):
    """
    Computes an STFT magnitude power spectrogram from an array of samples
    already cut to blocks.
    Input:
    - x - time-domain samples - array of shape (blocks, block_size)
    - w - window - array of shape (block_size)
    - to_log - indicates whether to scale the
    Output: spectrogram
    """
    X = magnitude_spectrum(x * w) ** 2
    if to_log:
        X = db_scale(X)
    return X

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
    """
    Real half of the spectrum. The DC term shared for positive and negative
    frequencies is halved.
    """
    N = X.shape[1]
    return np.hstack([0.5 *  X[:, :1], X[:, 1:N//2]])

def create_window(size):
    """
    A normalized Hanning window of given size. Useful for analyzing sinusoidal
    signals. It's normalized so that it has energy equal to its length, and mean
    power equal to 1.0.
    """
    w = scipy.hanning(size)
    w = w / mean_power(w)
    return w

def db_scale(magnitude_spectrum):
    """
    Transform linear magnitude to dbFS (full-scale).
    """
    # min_amplitude = 1e-6
    # threshold = -np.log10(min_amplitude)
    # return ((threshold + np.log10(np.maximum(min_amplitude, magnitude_spectrum))) / threshold)
    return 20 * np.log10(np.maximum(1e-6, magnitude_spectrum))

# -- extras --

def energy_weighted_spectrum(x):
    N = x.shape[-1]
    X = np.fft.fft(x)
    # np.allclose(energy(abs(X) / math.sqrt(N)), energy(x))
    # np.allclose(energy(abs(X[:N//2]) / math.sqrt(N//2)), energy(x))
    return abs(X) / math.sqrt(N)

def fftfreqs(block_size, fs):
    return np.fft.fftfreq(block_size, 1/fs)[:block_size // 2]

def inverse_spectrum(spectrum, window):
    '''
    inverse_spectrum(np.fft.fft(x * window), window) == x
    '''
    return np.real(np.fft.ifft(spectrum)) / window
