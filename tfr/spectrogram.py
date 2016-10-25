import numpy as np
import math
import scipy

from .features import mean_power
from .analysis import read_blocks

def spectrogram(filename, block_size=2048, hop_size=512, to_log=True):
    """
    Computes an STFT magnitude power spectrogram from an audio file.
    Returns: spectrogram, audio_samples, block_times
    """
    x, times, fs = read_blocks(filename, block_size, hop_size, mono_mix=True)
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

    For signal in range [-1., 1.] the output range is [0., 1.].

    The energy is not preserved, it's scaled down
    (energy_out = energy_in / (N//2)).
    '''
    X = np.fft.fft(x)
    Xr = positive_freq_magnitudes(X)
    N = Xr.shape[-1]
    return abs(Xr) / N

def select_positive_freq_fft(X):
    """
    Select the positive frequency part of the spectrum in a spectrogram.
    """
    N = X.shape[1]
    return X[:, :N//2]

# TODO: we should probably multiply the whole result by 2, to conserve energy
def positive_freq_magnitudes(X):
    """
    Select magnitudes from positive-frequency half of the spectrum in a
    spectrogram. The DC term shared for positive and negative frequencies is
    halved.

    Note this is not a complete information to reconstruct the full spectrum,
    since we throw away the bin at the negative Nyquist frequency (index N/2+1).
    """
    X_pos = select_positive_freq_fft(X).copy()
    X_pos[:, 0] *= 0.5
    return X_pos

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

    For input range [0, 1] the output range is [-120, 0].
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
    """
    Positive FFT frequencies from DC (incl.) until Nyquist (excl.).
    The size of half of the FTT size.
    """
    return np.fft.fftfreq(block_size, 1/fs)[:block_size // 2]

def inverse_spectrum(spectrum, window):
    '''
    inverse_spectrum(np.fft.fft(x * window), window) == x
    '''
    return np.real(np.fft.ifft(spectrum)) / window
