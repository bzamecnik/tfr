"""
An example showing analysis of multicomponent sound with and without time or
frequency reassignment. With both TF-reassignment the resulting spectrogram is
much sharper altough it still cannot resolve inseparable places.
"""

import numpy as np
import soundfile as sf
import tfr

def sample_time(since, until, fs=44100.):
    """
    Generates time sample in given interval [since; until]
    with given sampling rate (fs).
    """
    return np.arange(since, until, 1. / fs)

def freq_mod_sine(t, carrier_freq, mod_freq, mod_amp):
    return np.sin(2 * np.pi * carrier_freq * t
        + mod_amp * np.sin(2 * np.pi * mod_freq * t))

def sinusoid(t, freq):
    """
    t - array of time samples
    freq - frequency - constant or array of same size as t
    """
    return np.sin(2 * np.pi * freq * t)

def linear_chirp(t, start_freq, end_freq):
    slope = (end_freq - start_freq) / (t[-1] - t[0])
    return np.sin(2 * np.pi * (start_freq * t + 0.5 * slope * t**2))

def generate_example_sound(fs=44100, duration=3, carrier_freq=2000, mod_freq=1, mod_amp=1000, click_freq=10):
    t = sample_time(0, duration, fs)

    # FM component
    x = freq_mod_sine(t, carrier_freq, mod_freq, mod_amp)
    # constant tone component
    x += sinusoid(t, carrier_freq)
    # linear chirps
    x += linear_chirp(t, carrier_freq-mod_amp, carrier_freq+mod_amp)
    x += linear_chirp(t, carrier_freq+mod_amp, carrier_freq-mod_amp)
    # sawtooth pulse-train component
    for i in range(duration*click_freq):
        idx = int(i * fs / click_freq)
        x[idx:idx+10] = 1
    sf.write('multicomponent.wav', x, fs)
    return x, fs

if __name__ == '__main__':
    x, fs = generate_example_sound()
    signal_frames = tfr.SignalFrames(x, sample_rate=fs, frame_size=4096, hop_size=512)

    def band_pass(X, y_range):
        return X[:, slice(*y_range)]

    y_range = (84, 287) # 900-3100 Hz

    # plain non-reassigned spectrogram
    tfr.plots.save_raw_spectrogram_bitmap('multicomponent_stft.png',
        band_pass(
            tfr.reassigned_spectrogram(
                signal_frames,
                reassign_time=False,
                reassign_frequency=False),
            y_range))

    # time-reassigned spectrogram
    tfr.plots.save_raw_spectrogram_bitmap('multicomponent_t.png',
        band_pass(
            tfr.reassigned_spectrogram(signal_frames, reassign_frequency=False),
            y_range))

    # frequency-reassigned spectrogram
    tfr.plots.save_raw_spectrogram_bitmap('multicomponent_f.png',
        band_pass(
            tfr.reassigned_spectrogram(signal_frames, reassign_time=False),
            y_range))

    # time-frequency reassigned spectrogram
    tfr.plots.save_raw_spectrogram_bitmap('multicomponent_tf.png',
        band_pass(
            tfr.reassigned_spectrogram(signal_frames),
            y_range))
