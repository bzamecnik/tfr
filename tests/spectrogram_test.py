import numpy as np
from tfr.features import energy, mean_power
from tfr.spectrogram import create_window

def test_window_should_be_normalized():
    def assert_ok(size):
        w = create_window(size)
        assert np.allclose(energy(w), len(w))
        assert np.allclose(mean_power(w), 1.0)

    for size in [16, 100, 512, 777, 4096]:
        yield assert_ok, size
