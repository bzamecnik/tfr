from __future__ import print_function, division

import math
import os

import numpy as np
import scipy

from .spectrogram import db_scale, positive_freq_magnitudes, \
    select_positive_freq_fft, fftfreqs, normalized_window, scale_magnitudes
from .signal import SignalFrames
from .tuning import PitchQuantizer, Tuning
from .plots import save_raw_spectrogram_bitmap


class LinearTransform():
    def __init__(self, positive_only=True):
        # range of normalized frequencies
        self.bin_range = (0, 0.5) if positive_only else (0, 1)

    def transform_freqs(self, X_inst_freqs, sample_rate):
        output_bin_count = X_inst_freqs.shape[1]
        X_y = X_inst_freqs
        return X_y, output_bin_count, self.bin_range


class PitchTransform():
    """
    Perform the proper quantization to pitch bins according to possible
    subdivision before the actual histogram computation. Still we need to
    move the quantized pitch value a bit from the lower bin edge to ensure
    proper floating point comparison. Note that the quantizer rounds values
    from both sides towards the quantized value, while histogram2d floors the
    values to the lower bin edge. The epsilon is there to prevent log of 0
    in the pitch to frequency transformation.

    bin_range: range of pitch bins (default: A0 27.5 Hz to E10 21096.16 Hz)
    """
    def __init__(self, bin_range=(-48, 67), bin_division=1, tuning=Tuning()):
        self.tuning = tuning
        self.bin_range = bin_range
        self.bin_division = bin_division

    def transform_freqs(self, X_inst_freqs, sample_rate):
        quantization_border = 1 / (2 * self.bin_division)
        pitch_quantizer = PitchQuantizer(self.tuning, bin_division=self.bin_division)
        eps = np.finfo(np.float32).eps
        # TODO: is it possible to quantize using relative freqs to avoid
        # dependency on the fs parameter?
        X_y = pitch_quantizer.quantize(np.maximum(sample_rate * X_inst_freqs, eps) + quantization_border)
        output_bin_count = (self.bin_range[1] - self.bin_range[0]) * self.bin_division
        return X_y, output_bin_count, self.bin_range


class Spectrogram():
    """
    Represents spectrogram information of a time-domain signal which can be used
    to compute various types of reassigned spectrograms, pitchgrams, etc.
    """
    def __init__(self, signal_frames, window=scipy.hanning, positive_only=True):
        """
        :param signal_frames: signal represented as SignalFrames instance
        :param window: STFT window function - produces 1D window which will
        be normalized
        """
        self.signal_frames = signal_frames

        x_frames = signal_frames.frames
        w = normalized_window(window(signal_frames.frame_size))

        # complex spectra of windowed blocks of signal - STFT
        self.X_complex = np.fft.fft(x_frames * w)
        # linear magnitude spectrogram
        self.X_mag = abs(self.X_complex) / self.X_complex.shape[1]

        # spectra of signal shifted in time

        # This fakes looking at the previous frame shifted by one sample.
        # In order to work only with one frame of size N and not N + 1, we fill the
        # missing value with zero. This should not introduce a large error, since the
        # borders of the amplitude frame will go to zero anyway due to applying a
        # window function in the STFT tranform.
        X_prev_time = np.fft.fft(shift_right(x_frames) * w)

        # spectra shifted in frequency
        X_prev_freq = shift_right(self.X_complex)

        # cross-spectra - ie. spectra of cross-correlation between the
        # respective time-domain signals
        X_cross_time = cross_spectrum(self.X_complex, X_prev_time)
        X_cross_freq = cross_spectrum(self.X_complex, X_prev_freq)

        # instantaneous frequency estimates
        # normalized frequencies in range [0.0, 1.0] - from DC to sample rate
        self.X_inst_freqs = estimate_instant_freqs(X_cross_time)
        # instantaneous group delay estimates
        # relative coordinates within the frame with range [-0.5, 0.5] where
        # 0.0 is the frame center
        self.X_group_delays = estimate_group_delays(X_cross_freq)

        if positive_only:
            self.X_mag = positive_freq_magnitudes(self.X_mag)
            self.X_complex, self.X_inst_freqs, self.X_group_delays = [
                select_positive_freq_fft(values) for values in
                [self.X_complex, self.X_inst_freqs, self.X_group_delays]
            ]

    def reassigned(
        self,
        output_frame_size=None, transform=LinearTransform(),
        reassign_time=True, reassign_frequency=True, magnitudes='power_db'):
        """
        Reassigned spectrogram requantized both in frequency and time.

        Note it is quantized into non-overlapping output time frames which may be
        of a different size than input time frames.

        transform - transforms the frequencies
        """

        if output_frame_size is None:
            output_frame_size = self.signal_frames.hop_size

        frame_size = self.signal_frames.frame_size
        fs = self.signal_frames.sample_rate

        frame_duration = frame_size / fs
        frame_center_time = frame_duration / 2
        # group delays are in range [-0.5, 0.5] - relative coordinates within the
        # frame where 0.0 is the frame center
        input_bin_count = self.X_inst_freqs.shape[1]

        eps = np.finfo(np.float32).eps
        X_time = np.tile(self.signal_frames.start_times + frame_center_time +
            eps, (input_bin_count, 1)).T
        if reassign_time:
            X_time += self.X_group_delays * frame_duration

        if reassign_frequency:
            X_y = self.X_inst_freqs
        else:
            X_y = np.tile(fftfreqs(frame_size, fs) / fs, (self.X_inst_freqs.shape[0], 1))

        X_y, output_bin_count, bin_range = transform.transform_freqs(X_y,
            self.signal_frames.sample_rate)
        frame_duration = frame_size / fs
        end_input_time = self.signal_frames.duration
        output_frame_count = int(math.ceil((end_input_time * fs) / output_frame_size))
        print('output_frame_count', output_frame_count)
        time_range = (0, output_frame_count * output_frame_size / fs)

        output_shape = (output_frame_count, output_bin_count)
        X_spectrogram, x_edges, y_edges = np.histogram2d(
            X_time.flatten(), X_y.flatten(),
            weights=self.X_mag.flatten(),
            range=(time_range, bin_range),
            bins=output_shape)

        X_spectrogram = scale_magnitudes(X_spectrogram, magnitudes)

        return X_spectrogram

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


def process_spectrogram(filename, frame_size, hop_size, output_frame_size):
    """
    Computes three types of spectrograms (normal, frequency reassigned,
    time-frequency reassigned) from an audio file and stores and image from each
    spectrogram into PNG file.
    """
    signal_frames = SignalFrames(filename, frame_size, hop_size, mono_mix=True)

    spectrogram = Spectrogram(signal_frames)

    image_filename = os.path.basename(filename).replace('.wav', '')

    # STFT on overlapping input frames
    X_stft = db_scale(spectrogram.X_mag ** 2)
    save_raw_spectrogram_bitmap(image_filename + '_stft_frames.png', X_stft)

    linear_transform = LinearTransform(positive_only=True)

    # STFT requantized to the output frames (no reassignment)
    X_stft_requantized = spectrogram.reassigned(output_frame_size,
        linear_transform,
        reassign_time=False, reassign_frequency=False)
    save_raw_spectrogram_bitmap(image_filename + '_stft_requantized.png', X_stft_requantized)

    # STFT reassigned in time and requantized to output frames
    X_reassigned_t = spectrogram.reassigned(output_frame_size,
        linear_transform,
        reassign_time=True, reassign_frequency=False)
    save_raw_spectrogram_bitmap(image_filename + '_reassigned_t.png', X_reassigned_t)

    # STFT reassigned in frequency and requantized to output frames
    X_reassigned_f = spectrogram.reassigned(output_frame_size,
        linear_transform,
        reassign_time=False, reassign_frequency=True)
    save_raw_spectrogram_bitmap(image_filename + '_reassigned_f.png', X_reassigned_f)

    # STFT reassigned both in time and frequency and requantized to output frames
    X_reassigned_tf = spectrogram.reassigned(output_frame_size,
        linear_transform,
        reassign_time=True, reassign_frequency=True)
    save_raw_spectrogram_bitmap(image_filename + '_reassigned_tf.png', X_reassigned_tf)

    pitch_transform = PitchTransform(bin_range=(-48, 67), bin_division=1)

    # TF-reassigned pitchgram
    X_pitchgram_tf = spectrogram.reassigned(output_frame_size,
        pitch_transform,
        reassign_time=True, reassign_frequency=True)
    save_raw_spectrogram_bitmap(image_filename + '_pitchgram_tf.png', X_pitchgram_tf)

    # T-reassigned pitchgram
    X_pitchgram_t = spectrogram.reassigned(output_frame_size,
        pitch_transform,
        reassign_time=True, reassign_frequency=False)
    save_raw_spectrogram_bitmap(image_filename + '_pitchgram_t.png', X_pitchgram_t)

    # F-reassigned pitchgram
    X_pitchgram_t = spectrogram.reassigned(output_frame_size,
        pitch_transform,
        reassign_time=False, reassign_frequency=True)
    save_raw_spectrogram_bitmap(image_filename + '_pitchgram_f.png', X_pitchgram_t)

    # non-reassigned pitchgram
    X_pitchgram = spectrogram.reassigned(output_frame_size,
        pitch_transform,
        reassign_time=False, reassign_frequency=False)
    save_raw_spectrogram_bitmap(image_filename + '_pitchgram_no.png', X_pitchgram)

def reassigned_spectrogram(signal_frames, output_frame_size=None, magnitudes='power_db',
    reassign_time=True, reassign_frequency=True):
    """
    From frames of audio signal it computes the frequency reassigned spectrogram
    requantized back to the original linear bins.

    Only the real half of spectrum is given.
    """
    return Spectrogram(signal_frames).reassigned(
        output_frame_size, LinearTransform(),
        reassign_time, reassign_frequency, magnitudes=magnitudes)

# [-48,67) -> [~27.5, 21096.2) Hz
def pitchgram(signal_frames, output_frame_size=None, bin_range=(-48, 67), bin_division=1, magnitudes='power_db'):
    """
    From frames of audio signal it computes the frequency reassigned spectrogram
    requantized to pitch bins (pitchgram).
    """
    return Spectrogram(signal_frames).reassigned(
        output_frame_size, PitchTransform(bin_range, bin_division), magnitudes=magnitudes)

if __name__ == '__main__':
    import sys
    process_spectrogram(filename=sys.argv[1], frame_size=4096, hop_size=1024, output_frame_size=1024)
