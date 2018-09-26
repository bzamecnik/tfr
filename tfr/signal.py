from __future__ import print_function, division

import math
import numpy as np
import soundfile as sf

class SignalFrames():
    """
    Represents frames of time-domain signal of regular size with possible
    overlap plus it's metadata.

    The signal can be read from an numpy array a file via the soundfile library.
    The input array can be of shape `(samples,)` or `(samples, channels)`. By
    default the signal is mixed to mono to shape `(samples,)`. This can be
    disabled by specifying `mono_mix=False`.

    It is split into frames of `frame_size`. In case `hop_size < frame_size` the
    frames are overlapping. When the last frame is not fully covered by the
    signal it's padded with zeros.

    When reading signal from a file the sample rate can usually determined
    automatically, otherwise you shoud provide `sample_rate`.

    Attributes:
    - `frames` - signal split to frame, shape `(frames, frame_size)`
    - `frame_size`
    - `hop_size`
    - `length` - length of the source signal (in samples)
    - `duration` - duration of the source signal (in seconds)
    - `start_times` - array of start times of each frame (in seconds)

    Example usage:

    ```
    signal_frames = SignalFrames('audio.flac', frame_size=4096, hop_size=1024)
    spectrogram = np.fft.fft(signal_frames.frames * window)
    ```

    :param source: source of the time-domain signal - numpy array, file name,
    file-like object
    :param frame_size: size of each frame (in samples)
    :param hop_size: hop between frame starts (in samples)
    :param sample_rate: sample rate (required when source is an array)
    :mono_mix: indicates that multi-channel signal should be mixed to mono
    (mean of all channels)
    """
    def __init__(self, source, frame_size=4096, hop_size=2048, sample_rate=None,
        mono_mix=True):
        if type(source) == np.ndarray:
            signal = source
            self.sample_rate = sample_rate
        else:
            signal, self.sample_rate = sf.read(source)

        if mono_mix:
            signal = self._to_mono(signal)

        self.frames = self._split_to_frames(signal, frame_size, hop_size)
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.length = len(signal)
        self.duration = self.length / self.sample_rate
        self.start_times = np.arange(0, self.duration, self.hop_size / self.sample_rate)

    def _split_to_frames(self, x, frame_size, hop_size):
        """
        Splits the input audio signal to frame of given size (in samples).
        Start position of each frame is determined by given hop size.
        The last frame is right zero-padded if needed.
        input:
        x - array-like representing the audio signal
        """
        if hop_size is None:
            hop_size = frame_size
        frame_count = int(math.ceil(len(x) / hop_size))
        def pad(x, size, value=0):
            padding_size = size - len(x)
            if padding_size:
                x = np.pad(x, (0, padding_size), 'constant', constant_values=(0, 0))
            return x
        frames = np.vstack(
            pad(x[start:start + frame_size], frame_size) \
            for start in np.arange(0, hop_size * frame_count, hop_size))
        return frames

    def _to_mono(self, samples):
        if samples.ndim == 1:
            return samples
        else:
            return samples.mean(axis=-1)

def mean_power(x_frames):
    return np.sqrt(np.mean(x_frames**2, axis=-1))

def power(x_frames):
    return np.sqrt(np.sum(x_frames**2, axis=-1))

def mean_energy(x_frames):
    """
    Example usage:

    import matplotlib.pyplot as plt
    import soundfile as sf
    from analysis import read_frames

    def analyze_mean_energy(file, frame_size=1024):
        frames, t, fs = read_frames(x, frame_size)
        y = mean_energy(frames)
        plt.semilogy(t, y)
        plt.ylim(0, 1)
    """
    return np.mean(x_frames**2, axis=-1)

def energy(x_frames):
    return np.sum(x_frames**2, axis=-1)
