import os
import numpy as np

from .files import load_wav
from .spectrogram import real_half, create_window
from .analysis import split_to_blocks
from .tuning import quantize_freqs_to_pitch_bins
from .plots import save_raw_spectrogram_bitmap

def cross_spectrum(spectrumA, spectrumB):
    '''
    Returns a cross-spectrum, ie. spectrum of cross-correlation of two signals.
    This result does not depend on the order of the arguments.
    Since we already have the spectra of signals A and B and and want the
    spectrum of their cross-correlation, we can replace convolution in time
    domain with multiplication in frequency domain.
    '''
    return spectrumA * spectrumB.conj()

def shift_right(values):
    '''
    Shifts the array to the right by one place, filling the empty values with
    zeros.
    TODO: use np.roll()
    '''
    return np.hstack([np.zeros((values.shape[0],1)), values[..., :-1]])

def shifted_amplitude_pair(amplitudes):
    '''
    Fakes looking at the previous frame shifted by one sample.
    In order to work only with one frame of size N and not N + 1, we fill the
    missing value with zero. This should not introduce a large error, since the
    borders of the amplitude frame will go to zero anyway due to applying a
    window function in the STFT tranform.
    Returns: (previous amplitudes, current amplitudes)
    '''
    prevAmplitudes = shift_right(amplitudes)
    return prevAmplitudes, amplitudes

def arg(crossSpectrum):
    return np.mod(np.angle(crossSpectrum) / (2 * np.pi), 1.0)

def estimate_instant_freqs(crossTimeSpectrum):
    '''
    Channelized instantaneous frequency - the vector of simultaneous
    instantaneous frequencies computed over a single frame of the digital
    short-time Fourier transform.
    Instantaneous frequency - derivative of phase by time.
    cif = angle(crossSpectrumTime) * sampleRate / (2 * pi)
    In this case the return value is normalized (not multiplied by sampleRate).
    Basically it is phase normalized to the [0.0; 1.0] interval,
    instead of absolute [0.0; sampleRate].
    '''
    return arg(crossTimeSpectrum)

def estimate_group_delays(crossFreqSpectrum):
    return 0.5 - arg(crossFreqSpectrum)

def open_file(filename, block_size, hop_size):
    song, fs = load_wav(filename)
    x, times = split_to_blocks(song, block_size, hop_size=hop_size)
    return x, times, fs

def compute_spectra(x, w):
    X = np.fft.fft(x * w)
    X_prev_time = np.fft.fft(shift_right(x) * w)
    X_prev_freq = shift_right(X)
    X_cross_time = cross_spectrum(X, X_prev_time)
    X_cross_freq = cross_spectrum(X, X_prev_freq)
    X_inst_freqs = estimate_instant_freqs(X_cross_time)
    X_group_delays = estimate_group_delays(X_cross_freq)
    return X, X_cross_time, X_cross_freq, X_inst_freqs, X_group_delays

def db_scale(magnitude_spectrum):
    # min_amplitude = 1e-6
    # threshold = -np.log10(min_amplitude)
    # return ((threshold + np.log10(np.maximum(min_amplitude, magnitude_spectrum))) / threshold)
    return 20 * np.log10(np.maximum(1e-6, magnitude_spectrum))

def requantize_f_spectrogram(X_cross, X_instfreqs, to_log=True):
    '''Only requantize by frequency'''
    X_reassigned = np.empty(X_cross.shape)
    N = X_cross.shape[1]
    magnitude_spectrum = abs(X_cross) / N
    weights = magnitude_spectrum
    for i in range(X_cross.shape[0]):
        X_reassigned[i, :] = np.histogram(X_instfreqs[i], N, range=(0,1), weights=weights[i])[0]
    X_reassigned = X_reassigned ** 2
    if to_log:
         X_reassigned = db_scale(X_reassigned)
    return X_reassigned

def requantize_tf_spectrogram(X_group_delays, X_inst_freqs, times, block_size, fs, weights=None):
    block_duration = block_size / fs
    block_center_time = block_duration / 2
    X_time = np.tile(times + block_center_time, (X_group_delays.shape[1], 1)).T \
        + X_group_delays * block_duration
    time_range = (times[0], times[-1] + block_duration)
    freq_range = (0, 1)
    bins = X_inst_freqs.shape

    counts, x_edges, y_edges = np.histogram2d(
        X_time.flatten(), X_inst_freqs.flatten(),
        weights=weights.flatten(),
        range=(time_range, freq_range),
        bins=bins)
    return counts, x_edges, y_edges

def process_spectrogram(filename, block_size, hop_size):
    x, times, fs = open_file(filename, block_size, hop_size)
    w = create_window(block_size)
    X, X_cross_time, X_cross_freq, X_inst_freqs, X_group_delays = compute_spectra(x, w)

    X_reassigned_f = requantize_f_spectrogram(X_cross_time, X_inst_freqs)
    # N = X_cross.shape[1]
    # magnitude_spectrum = abs(X_cross_time) / N
    # weights = db_scale(magnitude_spectrum)
    X_magnitudes = abs(X_cross_time) / X.shape[1]
    weights = X_magnitudes
    X_reassigned_tf = requantize_tf_spectrogram(X_group_delays, X_inst_freqs, times, block_size, fs, weights)[0]
    X_reassigned_tf = db_scale(X_reassigned_tf ** 2)
    image_filename = os.path.basename(filename).replace('.wav', '.png')
    save_raw_spectrogram_bitmap('reassigned_f_' + image_filename, real_half(X_reassigned_f))
    save_raw_spectrogram_bitmap('reassigned_tf_' + image_filename, real_half(X_reassigned_tf))
    save_raw_spectrogram_bitmap('normal_' + image_filename, real_half(X_magnitudes))

#     X_time = X_group_delays + np.tile(np.arange(X.shape[0]).reshape(-1, 1), X.shape[1])
#     idx = (abs(X).flatten() > 10) & (X_inst_freqs.flatten() < 0.5)
#     plt.scatter(X_time.flatten()[idx], X_inst_freqs.flatten()[idx], alpha=0.1)
#     plt.savefig('scatter_' + image_filename)

def reassigned_spectrogram(x, w, to_log=True):
    X, X_cross_time, X_cross_freq, X_inst_freqs, X_group_delays = compute_spectra(x, w)
    X_reassigned_f = requantize_f_spectrogram(X_cross_time, X_inst_freqs, to_log)
    return real_half(X_reassigned_f)

def chromagram(x, w, fs, bin_range=(-48, 67), bin_division=1, to_log=True):
    "complete reassigned spectrogram with requantization to pitch bins"
    # TODO: better give frequency range
    X, X_cross_time, X_cross_freq, X_inst_freqs, X_group_delays = compute_spectra(x, w)
    n_blocks, n_freqs = X_cross_time.shape
    X_mag = abs(X_cross_time) / n_freqs
    weights = real_half(X_mag).flatten()
    eps = np.finfo(np.float32).eps
    pitch_bins = quantize_freqs_to_pitch_bins(np.maximum(fs * real_half(X_inst_freqs), eps), bin_division=bin_division).flatten()
    X_chromagram = np.histogram2d(
        np.repeat(np.arange(n_blocks), n_freqs / 2),
        pitch_bins,
        bins=(np.arange(n_blocks + 1),
              np.arange(bin_range[0], bin_range[1] + 1, 1 / bin_division)),
        weights=weights
    )[0]
    X_chromagram = X_chromagram ** 2
    if to_log:
        X_chromagram = db_scale(X_chromagram)
    return X_chromagram

def test_cross_spectrum():
    a = np.array([1j, 1+3j])
    b = np.array([2, 4j])
    c = np.array([-2j, 12+4j])
    assert_array_equals(cross_spectrum(a, b), c)

def test_shifted_amplitude_pair():
    actual = shifted_amplitude_pair(np.array([1,2,3]))
    assert_array_equals(actual[0], np.array([0, 1, 2]))
    assert_array_equals(actual[1], np.array([1, 2, 3]))

def assert_array_equals(a, b):
    assert (a == b).all()

if __name__ == '__main__':
    import sys
    process_spectrogram(filename=sys.argv[1], block_size=2048, hop_size=512)
