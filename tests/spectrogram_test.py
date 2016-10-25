import numpy as np
import os
from tfr.analysis import read_blocks
from tfr.features import energy, mean_power
from tfr.spectrogram import create_window, stft_spectrogram

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def test_window_should_be_normalized():
    def assert_ok(size):
        w = create_window(size)
        assert np.allclose(energy(w), len(w))
        assert np.allclose(mean_power(w), 1.0)

    for size in [16, 100, 512, 777, 4096]:
        yield assert_ok, size

def test_spectrogram_db_magnituds_should_be_in_proper_range():
    block_size = 4096
    audio_file = os.path.join(DATA_DIR, 'she_brings_to_me.wav')
    x_blocks, x_times, fs = read_blocks(audio_file, block_size=block_size)
    w = create_window(block_size)
    X = stft_spectrogram(x_blocks, w, to_log=True)
    assert np.all(X >= -120), 'min value: %f should be >= -120' % X.min()
    assert np.all(X <= 0), 'max value: %f should be <= 0' % X.max()

