from scipy.io import wavfile
import numpy as np
import subprocess
from scipy.signal import hilbert, chirp

from tuning import pitch_to_freq

def sample_time(since, until, fs=44100.):
    '''
    Generates time sample in given interval [since; until]
    with given sampling rate (fs).
    '''
    return np.arange(since, until, 1. / fs)

def sine(samples, freq=1., amplitude=1., phase=0.):
    '''
    Samples the sine function given the time samples,
    frequency (Hz), amplitude and phase [0; 2 * np.pi).
    '''
    print(freq)
    return amplitude * np.sin(2 * np.pi * freq * samples + phase)

def white_noise(samples, amplitude=1.):
    return amplitude * np.random.rand(*t.shape)

def normalize(samples):
    max_value = np.max(np.abs(samples))
    return samples / max_value if max_value != 0 else samples

def save_wav(samples, filename, fs=44100, normalize=False, factor=((2**15))-1):
    '''
    Saves samples in given sampling frequency to a WAV file.
    Samples are assumed to be in the [-1; 1] range and converted
    to signed 16-bit integers.
    '''
    samples = normalize(samples) if normalize else samples
    wavfile.write(filename, fs, np.int16(samples * factor))

def load_wav(filename, factor=(1 / (((2**15)) - 1))):
    '''
    Reads samples from a WAV file.
    Samples are assumed to be signed 16-bit integers and
    are converted to [-1; 1] range.
    It returns a tuple of sampling frequency and actual samples.
    '''
    fs, samples = wavfile.read(filename)
    samples = samples * factor
    return fs, samples

def play(filename):
    subprocess.call(['afplay', filename])

def generate_and_play(func, filename='test.wav', duration=1.,
    normalize=True, fade_ends=True, fade_length=100):
    t = sample_time(0, duration)
    samples = func(t)
    if fade_ends:
        samples = fade(samples, fade_length)
    save_wav(samples, filename, normalize=normalize)
    play(filename)
    return t, samples

def amplitude_envelope(x):
    return abs(hilbert(x))

def quadratic_fade_in(length):
    '''Quadratic fade-in window [0; 1] of given length.'''
    return 1 - (np.linspace(-1, 0, length) ** 2)

def fade(samples, n, fade_in=True, fade_out=True):
    x = np.copy(samples)
    in_window = quadratic_fade_in(n)
    if fade_in:
        x[:n] = in_window * x[:n]
    if fade_out:
        out_window = in_window[::-1]
        x[-n:] = out_window * x[-n:]
    return x

if __name__ == '__main__':

    # plain 440 Hz A for 1 second
    generate_and_play(lambda t: sine(t, 440))
    
    # 1 Hz dissonance
    generate_and_play(lambda t:
        np.sum(sine(t, f) for f in (440, 441)), duration=3)

    # 10 Hz dissonance
    generate_and_play(lambda t:
        np.sum(sine(t, 440 + 10 * i) for i in range(0, 2)), duration=3)

    # 10 harmonics with same amplitude
    generate_and_play(lambda t:
        np.sum(sine(t, 440 * (i + 1)) for i in range(0, 10)))

    # C-G fifth
    generate_and_play(lambda t:
        np.sum(sine(t, pitch_to_freq(i)) for i in (0, 4, 7)))
    
    # C major chord
    generate_and_play(lambda t:
        np.sum(sine(t, pitch_to_freq(i)) for i in (0, 4, 7)))

    # chirp signal - non-constant frequency
    
    # linear chirp
    generate_and_play(lambda t: chirp(t, 440, 1, 880))
    
    # constant 440 + raising ramp from 430 to 450
    generate_and_play(lambda t: sine(t, 440) + chirp(t, 440-10, 4, 440+10), duration=4)
    
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
    generate_and_play(lambda t: np.sum(sine(t, f) for f in (440, 445)), duration=3)

    # derivative of the amplitude envelope
    plot(t[:-1], abs(np.diff(abs(hilbert(x))))
