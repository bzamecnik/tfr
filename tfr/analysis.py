import numpy as np
import math


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
    times = split_block_times(len(x), fs, hop_size)
    return blocks, times

def split_block_times(N, fs, hop_size):
    return np.arange(0, N/fs, hop_size/fs)

def to_mono(samples):
    if samples.ndim == 1:
        return samples
    else:
        return samples.mean(axis=-1)
