from __future__ import print_function, division

import numpy as np
import math
import scipy

from .signal import mean_power
from .signal import SignalFrames

def spectrogram(filename, frame_size=2048, hop_size=512, magnitudes='power_db'):
    """
    Computes an STFT magnitude power spectrogram from an audio file.
    Returns: spectrogram, audio_samples, frame_times
    """
    signal_frames = SignalFrames(filename, frame_size, hop_size, mono_mix=True)
    x = signal_frames.frames
    times = signal_frames.start_times
    w = create_window(frame_size)
    X = stft_spectrogram(x, w, magnitudes)
    return X, x, times

def stft_spectrogram(x, w, magnitudes):
    """
    Computes an STFT magnitude power spectrogram from an array of samples
    already cut to frames.
    Input:
    - x - time-domain samples - array of shape (frames, frame_size)
    - w - window - array of shape (frame_size)
    - magnitudes - indicates whether to scale the
    Output: spectrogram
    """
    X = magnitude_spectrum(x * w) ** 2
    if magnitudes:
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
    signals.
    """
    return normalized_window(scipy.hanning(size))

def normalized_window(w):
    """
    Normalizes an FFT window so that it has energy equal to its length, and mean
    power equal to 1.0.
    """
    return w / mean_power(w)

def db_scale(magnitude_spectrum, normalized=False):
    """
    Transform linear magnitude to dbFS (full-scale) [-120, 0] (for input range
    [0.0, 1.0]) which can be optionally normalized to [0.0, 1.0].
    """
    scaled = 20 * np.log10(np.maximum(1e-6, magnitude_spectrum))

    # map from raw dB [-120.0, 0] to [0.0, 1.0]
    if normalized:
        scaled = (scaled / 120) + 1
    return scaled

def scale_magnitudes(X_mag, transform):
    if transform == 'linear':
        return X_mag
    elif transform == 'power':
        return X_mag ** 2
    elif transform == 'power_db':
        return db_scale(X_mag ** 2)
    elif transform == 'power_db_normalized':
        return db_scale(X_mag ** 2, normalized=True)
    else:
        raise ValueError('Unknown magnitude scaling transform ' + transform)

# -- extras --

def energy_weighted_spectrum(x):
    N = x.shape[-1]
    X = np.fft.fft(x)
    # np.allclose(energy(abs(X) / math.sqrt(N)), energy(x))
    # np.allclose(energy(abs(X[:N//2]) / math.sqrt(N//2)), energy(x))
    return abs(X) / math.sqrt(N)

def fftfreqs(frame_size, fs):
    """
    Positive FFT frequencies from DC (incl.) until Nyquist (excl.).
    The size of half of the FTT size.
    """
    return np.fft.fftfreq(frame_size, 1/fs)[:frame_size // 2]

def inverse_spectrum(spectrum, window):
    '''
    inverse_spectrum(np.fft.fft(x * window), window) == x
    '''
    return np.real(np.fft.ifft(spectrum)) / window
