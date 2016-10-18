import numpy as np
from scipy.io import wavfile

def normalize(samples):
    max_value = np.max(np.abs(samples))
    return samples / max_value if max_value != 0 else samples

def save_wav(samples, filename, fs=44100, should_normalize=False, factor=((2**15))-1):
    '''
    Saves samples in given sampling frequency to a WAV file.
    Samples are assumed to be in the [-1; 1] range and converted
    to signed 16-bit integers.
    '''
    samples = normalize(samples) if should_normalize else samples
    wavfile.write(filename, fs, np.int16(samples * factor))

def load_wav(filename, factor=(1 / (((2**15)) - 1)), mono_mix=True):
    '''
    Reads samples from a WAV file.
    Samples are assumed to be signed 16-bit integers and
    are converted to [-1; 1] range.
    It returns a tuple of sampling frequency and actual samples.
    '''
    fs, samples = wavfile.read(filename)
    samples = samples * factor
    if mono_mix:
        samples = to_mono(samples)
    return samples, fs

def to_mono(samples):
    if samples.ndim == 1:
        return samples
    else:
        return samples.mean(axis=-1)
