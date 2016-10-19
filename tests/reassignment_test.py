import numpy as np

from tfr.reassignment import cross_spectrum, shifted_amplitude_pair


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
