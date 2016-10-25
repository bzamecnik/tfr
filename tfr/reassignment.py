import os
import numpy as np

from .spectrogram import db_scale, positive_freq_magnitudes, create_window
from .analysis import read_blocks
from .tuning import PitchQuantizer, Tuning
from .plots import save_raw_spectrogram_bitmap

def cross_spectrum(spectrumA, spectrumB):
    """
    Returns a cross-spectrum, ie. spectrum of cross-correlation of two signals.
    This result does not depend on the order of the arguments.
    Since we already have the spectra of signals A and B and and want the
    spectrum of their cross-correlation, we can replace convolution in time
    domain with multiplication in frequency domain.
    """
    return spectrumA * spectrumB.conj()

def shift_right(values):
    """
    Shifts the array to the right by one place, filling the empty values with
    zeros.
    TODO: use np.roll()
    """
    # TODO: this fails for 1D input array!
    return np.hstack([np.zeros((values.shape[0], 1)), values[..., :-1]])

def arg(values):
    """
    Argument (angle) of complex numbers wrapped and scaled to [0.0, 1.0].

    input: an array of complex numbers
    output: an array of real numbers of the same shape

    np.angle() returns values in range [-np.pi, np.pi].
    """
    return np.mod(np.angle(values) / (2 * np.pi), 1.0)

def estimate_instant_freqs(crossTimeSpectrum):
    """
    Channelized instantaneous frequency - the vector of simultaneous
    instantaneous frequencies computed over a single frame of the digital
    short-time Fourier transform.

    Instantaneous frequency - derivative of phase by time.

    cif = angle(crossSpectrumTime) * sampleRate / (2 * pi)

    In this case the return value is normalized (not multiplied by sampleRate)
    to the [0.0; 1.0] interval, instead of absolute [0.0; sampleRate].
    """
    return arg(crossTimeSpectrum)

def estimate_group_delays(crossFreqSpectrum):
    "range: [-0.5, 0.5]"
    return 0.5 - arg(crossFreqSpectrum)

def compute_spectra(x, w):
    """
    This computes all the spectra needed for reassignment as well as estimates
    of instantaneous frequency and group delay.

    Input:
    - x - an array of time blocks
    - w - 1D normalized window of the same size as x.shape[0]
    """
    # normal spectrum (with a window)
    X = np.fft.fft(x * w)
    # spectrum of signal shifted in time
    # This fakes looking at the previous frame shifted by one sample.
    # In order to work only with one frame of size N and not N + 1, we fill the
    # missing value with zero. This should not introduce a large error, since the
    # borders of the amplitude frame will go to zero anyway due to applying a
    # window function in the STFT tranform.
    X_prev_time = np.fft.fft(shift_right(x) * w)
    # spectrum shifted in frequency
    X_prev_freq = shift_right(X)
    X_cross_time = cross_spectrum(X, X_prev_time)
    X_cross_freq = cross_spectrum(X, X_prev_freq)
    X_inst_freqs = estimate_instant_freqs(X_cross_time)
    X_group_delays = estimate_group_delays(X_cross_freq)
    return X, X_cross_time, X_cross_freq, X_inst_freqs, X_group_delays

def requantize_f_spectrogram(X, X_instfreqs, to_log=True):
    """Spectrogram requantized only in frequency"""
    X_reassigned = np.empty(X.shape)
    N = X.shape[1]
    magnitude_spectrum = abs(X) / N
    weights = magnitude_spectrum
    for i in range(X.shape[0]):
        X_reassigned[i, :] = np.histogram(X_instfreqs[i], N, range=(0,1), weights=weights[i])[0]
    X_reassigned = X_reassigned ** 2
    if to_log:
         X_reassigned = db_scale(X_reassigned)
    return X_reassigned

def requantize_tf_spectrogram(X_group_delays, X_inst_freqs, times, block_size, fs, weights=None):
    """Spectrogram requantized both in frequency and time"""
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
    """
    Computes three types of spectrograms (normal, frequency reassigned,
    time-frequency reassigned) from an audio file and stores and image from each
    spectrogram into PNG file.
    """
    x, times, fs = read_blocks(filename, block_size, hop_size, mono_mix=True)
    w = create_window(block_size)
    X, X_cross_time, X_cross_freq, X_inst_freqs, X_group_delays = compute_spectra(x, w)

    X_reassigned_f = requantize_f_spectrogram(X, X_inst_freqs)
    X_magnitudes = abs(X) / X.shape[1]
    weights = X_magnitudes
    X_reassigned_tf = requantize_tf_spectrogram(X_group_delays, X_inst_freqs, times, block_size, fs, weights)[0]
    X_reassigned_tf = db_scale(X_reassigned_tf ** 2)
    image_filename = os.path.basename(filename).replace('.wav', '.png')
    save_raw_spectrogram_bitmap('reassigned_f_' + image_filename, positive_freq_magnitudes(X_reassigned_f))
    save_raw_spectrogram_bitmap('reassigned_tf_' + image_filename, positive_freq_magnitudes(X_reassigned_tf))
    save_raw_spectrogram_bitmap('normal_' + image_filename, positive_freq_magnitudes(X_magnitudes))

#     X_time = X_group_delays + np.tile(np.arange(X.shape[0]).reshape(-1, 1), X.shape[1])
#     idx = (abs(X).flatten() > 10) & (X_inst_freqs.flatten() < 0.5)
#     plt.scatter(X_time.flatten()[idx], X_inst_freqs.flatten()[idx], alpha=0.1)
#     plt.savefig('scatter_' + image_filename)

def reassigned_spectrogram(x, w, to_log=True):
    """
    From blocks of audio signal it computes the frequency reassigned spectrogram
    requantized back to the original linear bins.

    Only the real half of spectrum is given.
    """
    # TODO: The computed arrays are symetrical (positive vs. negative freqs).
    # We should only use one half.
    X, X_cross_time, X_cross_freq, X_inst_freqs, X_group_delays = compute_spectra(x, w)
    X_reassigned_f = requantize_f_spectrogram(X, X_inst_freqs, to_log)
    return positive_freq_magnitudes(X_reassigned_f)

def chromagram(x, w, fs, bin_range=(-48, 67), bin_division=1, to_log=True):
    """
    From blocks of audio signal it computes the frequency reassigned spectrogram
    requantized to pitch bins (chromagram).
    """
    # TODO: better give frequency range
    X, X_cross_time, X_cross_freq, X_inst_freqs, X_group_delays = compute_spectra(x, w)
    n_blocks, n_freqs = X_cross_time.shape
    X_mag = abs(X) / n_freqs
    weights = positive_freq_magnitudes(X_mag).flatten()
    eps = np.finfo(np.float32).eps
    pitch_quantizer = PitchQuantizer(Tuning(), bin_division=bin_division)
    # TODO: is it possible to quantize using relative freqs to avoid
    # dependency on the fs parameter?
    pitch_bins = pitch_quantizer.quantize(np.maximum(fs * positive_freq_magnitudes(X_inst_freqs), eps)).flatten()
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


# unused - range of bins for the chromagram
def pitch_bin_range(pitch_start, pitch_end, tuning):
    """
    Generates a range of pitch bins and their frequencies.
    """
    # eg. [-48,67) -> [~27.5, 21096.2) Hz
    pitch_range = np.arange(pitch_start, pitch_end)
    bin_center_freqs = np.array([tuning.pitch_to_freq(f) for f in pitch_range])
    return pitch_range, bin_center_freqs

if __name__ == '__main__':
    import sys
    process_spectrogram(filename=sys.argv[1], block_size=2048, hop_size=512)
