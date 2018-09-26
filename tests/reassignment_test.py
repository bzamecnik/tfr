from __future__ import print_function, division

import numpy as np
import os

from tfr import SignalFrames, Tuning, pitchgram, reassigned_spectrogram
from tfr.reassignment import shift_right, arg

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def test_shift_right():
    assert np.allclose(shift_right(np.array([[1, 2, 3]])), np.array([0, 1, 2]))

def test_pitchgram_on_single_tone_should_have_peak_at_that_tone():
    pitch = 12 + 7 # G5
    f = Tuning().pitch_to_freq(pitch)
    fs = 44100
    x = sine(sample_time(0, 1, fs=fs), freq=f)
    frame_size = 4096
    hop_size = 2048
    output_frame_size = hop_size
    signal_frames = SignalFrames(x, frame_size, hop_size, sample_rate=fs, mono_mix=True)
    bin_range = [-48, 67]
    x_pitchgram = pitchgram(signal_frames,
        output_frame_size, magnitudes='power_db', bin_range=bin_range, bin_division=1)

    max_bin_expected = pitch - bin_range[0]
    max_bin_actual = x_pitchgram.mean(axis=0).argmax()

    assert x_pitchgram.shape == (22, 115), x_pitchgram.shape
    assert max_bin_actual == max_bin_expected

def test_arg():
    values = np.array([-5.-1.j, -1.-5.j,  2.-2.j,  3.+4.j,  2.+0.j,  2.-5.j, -3.-3.j,
       -3.+1.j, -2.-5.j,  0.+2.j])
    args = arg(values)
    expected_args=np.array([0.53141648, 0.71858352, 0.875 , 0.14758362, 0.,
        0.81055947, 0.625 , 0.44879181, 0.68944053, 0.25])
    assert np.allclose(args, expected_args)

def test_reassigned_spectrogram_values_should_be_in_proper_range():
    frame_size = 4096
    hop_size = frame_size
    output_frame_size = 1024
    audio_file = os.path.join(DATA_DIR, 'she_brings_to_me.wav')
    signal_frames = SignalFrames(audio_file, frame_size, hop_size, mono_mix=True)
    X_r = reassigned_spectrogram(signal_frames, output_frame_size, magnitudes='power_db')
    assert np.all(X_r >= -120), 'min value: %f should be >= -120' % X_r.min()
    assert np.all(X_r <= 0), 'max value: %f should be <= 0' % X_r.max()

def test_reassigned_pitchgram_values_should_be_in_proper_range():
    frame_size = 4096
    hop_size = frame_size
    output_frame_size = 1024
    audio_file = os.path.join(DATA_DIR, 'she_brings_to_me.wav')
    signal_frames = SignalFrames(audio_file, frame_size, hop_size, mono_mix=True)
    X_r = pitchgram(signal_frames, output_frame_size, magnitudes='power_db')
    assert np.all(X_r >= -120), 'min value: %f should be >= -120' % X_r.min()
    assert np.all(X_r <= 0), 'max value: %f should be <= 0' % X_r.max()

# --- helper functions ---

def sample_time(since, until, fs=44100.):
    '''
    Generates time sample in given interval [since; until]
    with given sampling rate (fs).
    '''
    return np.arange(since, until, 1. / fs)

def sine(t, freq=1., amplitude=1., phase=0.):
    '''
    Samples the sine function given the time samples t,
    frequency (Hz), amplitude and phase [0; 2 * np.pi).
    '''
    return amplitude * np.sin(2 * np.pi * freq * t + phase)
