import numpy as np
from scipy.signal import chirp

from tuning import pitch_to_freq
from files import save_wav

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

def generate_and_save(func, filename='test.wav', duration=1.,
    normalize=True, fade_ends=True, fade_length=100):
    t = sample_time(0, duration)
    samples = func(t)
    if fade_ends:
        samples = fade(samples, fade_length)
    save_wav(samples, filename, should_normalize=normalize)
    return t, samples

if __name__ == '__main__':
    from playback import generate_and_play

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


    # Plomp-Levelt roughness maximum at the fifth
    generate_and_play(lambda t:
        np.sum(sine(t, f) for f in (36, 54)))
