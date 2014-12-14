import numpy as np
from scipy.signal import hilbert

from synthesis import sample_time, sine
from playback import generate_and_play

def amplitude_envelope(x):
    return abs(hilbert(x))

if __name__ == '__main__':
    
    # Dissonance of two sines f_1 and f_2 is like
    # amplitude modulation of abs(f_1 - f_2) onto mean(f_1, f_2)
    # The amplitude envelope can be obtained as the
    # absolute value of the analytical signal (x + i * h(x))
    # where h is the Hilbert transform.
    # abs(scipy.signal.hilbert(x))
    # It corresponds to the AM demodulation.
    # The beating frequency is 2 * (f_1 - f_2), since its absolute
    # value has twice higher the period.
    t = sample_time(0, 1, 200)
    x = sine(t, 10)  + sine(t, 12)
    e = amplitude_envelope(x)
    plot(t, x)
    plot(t, e)
    # generate_and_play(lambda t: np.sum(sine(t, f) for f in (440, 445)), duration=3)

    # derivative of the amplitude envelope
    # plot(t[:-1], abs(np.diff(abs(hilbert(x)))))
