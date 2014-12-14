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

def save_wav(samples, filename, fs=44100, normalize=False, factor=((2**15))-1):
    samples = samples / np.max(np.abs(samples)) if normalize else samples
    wavfile.write(filename, fs, np.int16(samples * factor))

def play(filename):
    subprocess.call(['afplay', filename])

def generate_and_play(func, duration=1.):
    filename = 'test.wav'
    t = sample_time(0, duration)
    samples = func(t)
    save_wav(samples, filename, normalize=True)
    play(filename)

def amplitude_envelope(x):
    return abs(hilbert(x))

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
