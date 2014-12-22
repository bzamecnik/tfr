import sys
sys.path.append('../..')
import numpy as np

from tuning import pitch_to_freq
from synthesis import sine, generate_and_save

def complex_tone(partials):
    # partial = ((freq0, amp0), (freq1, amp1), ...)
    return lambda t: np.sum(sine(t, freq=freq, amplitude=amp) for (freq, amp) in partials)

def dissonance(base, difference):
    '''Returns the timbre composed of two sines of unit amplitude.'''
    return ((base, 1), (base + difference, 1))

def uniform_random(size, a, b):
    return a + (b - a) * np.random.random(size)

if __name__ == '__main__':
    numpy.random.seed(42)
    
    log_diff_range = (-1, 4)
    
    sample_count = 100
    log_diffs = uniform_random(sample_count, *log_diff_range)
    
    base = 440
    for log_diff in log_diffs:
        diff = 10 ** log_diff
        generate_and_save(complex_tone(dissonance(base, diff)),
            filename="%s_%s.wav" % (base, diff), duration=3)
