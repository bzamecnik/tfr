import numpy as np
from scipy.signal import hilbert
import math

from synthesis import sample_time, sine
from playback import generate_and_play

def amplitude_envelope(x):
    return abs(hilbert(x))

def split_to_blocks(x, block_size=1024, hop_size=None, fs=44100):
    '''
    Splits the input audio signal to block of given size (in samples).
    Start position of each block is determined by given hop size.
    The last block is right zero-padded if needed.
    input:
    x - array-like representing the audio signal
    '''
    if hop_size is None:
        hop_size = block_size
    block_count = math.ceil(len(x) / hop_size)
    def pad(x, size, value=0):
        padding_size = size - len(x)
        if padding_size:
            x = np.pad(x, (0, padding_size), 'constant', constant_values=(0, 0))
        return x
    blocks = np.vstack(
        pad(x[start:start + block_size], block_size) \
        for start in range(0, hop_size * block_count, hop_size))
    times = np.arange(0, len(x)/fs, hop_size/fs)
    return blocks, times

def test_split_to_blocks():
    assert np.array([
        [ 0,  1,  2,  3,  4,  5,  6,  7],
        [ 6,  7,  8,  9, 10, 11, 12, 13],
        [12, 13, 14, 15, 16, 17, 18, 19],
        [18, 19, 20, 21, 22,  0,  0,  0],
    ]) == split_to_blocks(np.arange(23), 8, 6)
        

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
