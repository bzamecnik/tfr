import os
import numpy as np

from .spectrogram import db_scale, positive_freq_magnitudes, create_window, \
    select_positive_freq_fft, fftfreqs
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

def compute_spectra(x, w, positive_only=True):
    """
    This computes all the spectra needed for reassignment as well as estimates
    of instantaneous frequency and group delay.

    Input:
    - x - an array of time blocks
    - w - 1D normalized window of the same size as x.shape[0]
    """
    # normal spectrum (with a window)
    X = np.fft.fft(x * w)
    X_mag = abs(X) / X.shape[1]
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

    if positive_only:
        X_mag = positive_freq_magnitudes(X_mag)
        X, X_cross_time, X_cross_freq, X_inst_freqs, X_group_delays = [
            select_positive_freq_fft(values) for values in
            [X, X_cross_time, X_cross_freq, X_inst_freqs, X_group_delays]
        ]

    return X, X_mag, X_cross_time, X_cross_freq, X_inst_freqs, X_group_delays

# deprecated
def requantize_f_spectrogram(X_mag, X_inst_freqs, to_log=True, positive_only=True):
    """
    Spectrogram requantized only in frequency.

    positive_only - indicates that input and output is the positive half of the
    spectrum
    """
    X_reassigned = np.empty(X_mag.shape)
    N = X_mag.shape[1]
    # range of normalized frequencies
    freq_range = (0, 0.5) if positive_only else (0, 1)
    for i in range(X_mag.shape[0]):
        X_reassigned[i, :] = np.histogram(X_inst_freqs[i], N, range=freq_range, weights=X_mag[i])[0]
    X_reassigned = X_reassigned ** 2
    if to_log:
        X_reassigned = db_scale(X_reassigned)
    return X_reassigned

def requantize_tf_spectrogram(X_group_delays, X_inst_freqs, times, block_size,
    output_frame_size, fs, weights=None, positive_only=True):
    """
    Spectrogram requantized both in frequency and time.

    Note it is quantized into non-overlapping output time frames which may be
    of a different size than input time frames.
    """
    block_duration = block_size / fs
    block_center_time = block_duration / 2
    # group delays are in range [-0.5, 0.5] - relative coordinates within the
    # block where 0.0 is the block center
    freq_bin_count = X_inst_freqs.shape[1]
    X_time = np.tile(times + block_center_time + np.finfo(np.float32).eps, (freq_bin_count, 1)).T
    if X_group_delays is not None:
        X_time += X_group_delays * block_duration
    end_input_time = times[-1] + block_duration
    output_frame_count = (end_input_time * fs) // output_frame_size
    time_range = (0, output_frame_count * output_frame_size / fs)
    # range of normalized frequencies
    freq_range = (0, 0.5) if positive_only else (0, 1)

    output_shape = (output_frame_count, freq_bin_count)
    counts, x_edges, y_edges = np.histogram2d(
        X_time.flatten(), X_inst_freqs.flatten(),
        weights=weights.flatten(),
        range=(time_range, freq_range),
        bins=output_shape)

    return counts, x_edges, y_edges

# TODO: deduplicate with requantize_tf_spectrogram
def requantize_tf_chromagram(X_group_delays, X_inst_freqs, times, block_size,
    output_frame_size, fs, weights=None, bin_range=(-48, 67), bin_division=1):
    block_duration = block_size / fs
    block_center_time = block_duration / 2
    # group delays are in range [-0.5, 0.5] - relative coordinates within the
    # block where 0.0 is the block center
    freq_bin_count = X_inst_freqs.shape[1]
    pitch_bin_count = (bin_range[1] - bin_range[0]) * bin_division

    X_time = np.tile(times + block_center_time + np.finfo(np.float32).eps, (freq_bin_count, 1)).T
    if X_group_delays is not None:
        X_time += X_group_delays * block_duration

    eps = np.finfo(np.float32).eps
    pitch_quantizer = PitchQuantizer(Tuning(), bin_division=bin_division)
    # TODO: is it possible to quantize using relative freqs to avoid
    # dependency on the fs parameter?
    pitch_bins = pitch_quantizer.quantize(np.maximum(fs * X_inst_freqs, eps)).flatten()

    end_input_time = times[-1] + block_duration
    output_frame_count = (end_input_time * fs) // output_frame_size
    time_range = (0, output_frame_count * output_frame_size / fs)
    # range of normalized frequencies

    output_shape = (output_frame_count, pitch_bin_count)
    counts, x_edges, y_edges = np.histogram2d(
        X_time.flatten(), pitch_bins.flatten(),
        weights=weights.flatten(),
        range=(time_range, bin_range),
        bins=output_shape)

    return counts, x_edges, y_edges

def process_spectrogram(filename, block_size, hop_size, output_frame_size):
    """
    Computes three types of spectrograms (normal, frequency reassigned,
    time-frequency reassigned) from an audio file and stores and image from each
    spectrogram into PNG file.
    """
    x, times, fs = read_blocks(filename, block_size, hop_size, mono_mix=True)
    w = create_window(block_size)
    X, X_mag, X_cross_time, X_cross_freq, X_inst_freqs, X_group_delays = compute_spectra(x, w)


    image_filename = os.path.basename(filename).replace('.wav', '')

    # STFT on overlapping input frames
    X_stft = db_scale(X_mag ** 2)
    save_raw_spectrogram_bitmap(image_filename + '_stft_frames.png', X_stft)

    X_bin_freqs = np.tile(fftfreqs(block_size, fs)/fs, (X_inst_freqs.shape[0], 1))

    # STFT requantized to the output frames (no reassignment)
    X_stft_requantized = requantize_tf_spectrogram(None, X_bin_freqs, times, block_size, output_frame_size, fs, X_mag)[0]
    X_stft_requantized = db_scale(X_stft_requantized ** 2)
    save_raw_spectrogram_bitmap(image_filename + '_stft_requantized.png', X_stft_requantized)

    # STFT reassigned in time and requantized to output frames
    X_reassigned_t = requantize_tf_spectrogram(X_group_delays, X_bin_freqs, times, block_size, output_frame_size, fs, X_mag)[0]
    X_reassigned_t = db_scale(X_reassigned_t ** 2)
    save_raw_spectrogram_bitmap(image_filename + '_reassigned_t.png', X_reassigned_t)

    # STFT reassigned in frequency and requantized to output frames
    X_reassigned_f = requantize_tf_spectrogram(None, X_inst_freqs, times, block_size, output_frame_size, fs, X_mag)[0]
    X_reassigned_f = db_scale(X_reassigned_f ** 2)
    save_raw_spectrogram_bitmap(image_filename + '_reassigned_f.png', X_reassigned_f)

    # STFT reassigned both in time and frequency and requantized to output frames
    X_reassigned_tf = requantize_tf_spectrogram(X_group_delays, X_inst_freqs, times, block_size, output_frame_size, fs, X_mag)[0]
    X_reassigned_tf = db_scale(X_reassigned_tf ** 2)
    save_raw_spectrogram_bitmap(image_filename + '_reassigned_tf.png', X_reassigned_tf)

    # chromagram
    X_chromagram = requantize_tf_chromagram(X_group_delays, X_inst_freqs, times, block_size, output_frame_size, fs, X_mag)[0]
    X_chromagram = db_scale(X_chromagram ** 2)
    save_raw_spectrogram_bitmap(image_filename + '_chromagram.png', X_chromagram)

def reassigned_spectrogram(x, w, to_log=True):
    """
    From blocks of audio signal it computes the frequency reassigned spectrogram
    requantized back to the original linear bins.

    Only the real half of spectrum is given.
    """
    # TODO: The computed arrays are symetrical (positive vs. negative freqs).
    # We should only use one half.
    X, X_mag, X_cross_time, X_cross_freq, X_inst_freqs, X_group_delays = compute_spectra(x, w)
    return requantize_f_spectrogram(X_mag, X_inst_freqs, to_log)

def chromagram(x, w, fs, bin_range=(-48, 67), bin_division=1, to_log=True):
    """
    From blocks of audio signal it computes the frequency reassigned spectrogram
    requantized to pitch bins (chromagram).
    """
    # TODO: better give frequency range
    X, X_mag, X_cross_time, X_cross_freq, X_inst_freqs, X_group_delays = compute_spectra(x, w)
    n_blocks, n_freqs = X.shape
    weights = X_mag.flatten()
    eps = np.finfo(np.float32).eps
    pitch_quantizer = PitchQuantizer(Tuning(), bin_division=bin_division)
    # TODO: is it possible to quantize using relative freqs to avoid
    # dependency on the fs parameter?
    pitch_bins = pitch_quantizer.quantize(np.maximum(fs * X_inst_freqs, eps)).flatten()
    X_chromagram = np.histogram2d(
        np.repeat(np.arange(n_blocks), n_freqs),
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
    process_spectrogram(filename=sys.argv[1], block_size=2048, hop_size=512, output_frame_size=512)
