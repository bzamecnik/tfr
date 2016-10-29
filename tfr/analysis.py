import math
import numpy as np
import soundfile as sf

def read_frames(filename, frame_size, hop_size=None, mono_mix=True):
    song, fs = sf.read(filename)
    if mono_mix:
        song = to_mono(song)
    x, times = split_to_frames(song, frame_size, hop_size=hop_size)
    return x, times, fs

def split_to_frames(x, frame_size=1024, hop_size=None, fs=44100):
    '''
    Splits the input audio signal to frame of given size (in samples).
    Start position of each frame is determined by given hop size.
    The last frame is right zero-padded if needed.
    input:
    x - array-like representing the audio signal
    '''
    if hop_size is None:
        hop_size = frame_size
    frame_count = math.ceil(len(x) / hop_size)
    def pad(x, size, value=0):
        padding_size = size - len(x)
        if padding_size:
            x = np.pad(x, (0, padding_size), 'constant', constant_values=(0, 0))
        return x
    frames = np.vstack(
        pad(x[start:start + frame_size], frame_size) \
        for start in range(0, hop_size * frame_count, hop_size))
    times = split_frame_times(len(x), fs, hop_size)
    return frames, times

def split_frame_times(N, fs, hop_size):
    return np.arange(0, N/fs, hop_size/fs)

def to_mono(samples):
    if samples.ndim == 1:
        return samples
    else:
        return samples.mean(axis=-1)
