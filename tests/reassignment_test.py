import numpy as np

from tfr.reassignment import shifted_amplitude_pair


def test_shifted_amplitude_pair():
    actual = shifted_amplitude_pair(np.array([[1, 2, 3]]))
    assert np.allclose(actual[0], np.array([0, 1, 2]))
    assert np.allclose(actual[1], np.array([1, 2, 3]))
