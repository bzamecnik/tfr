import subprocess

from synthesis import sample_time, fade
from files import save_wav

def play(filename):
    subprocess.call(['afplay', filename])

def generate_and_play(func, filename='test.wav', duration=1.,
    normalize=True, fade_ends=True, fade_length=100):
    t = sample_time(0, duration)
    samples = func(t)
    if fade_ends:
        samples = fade(samples, fade_length)
    save_wav(samples, filename, should_normalize=normalize)
    play(filename)
    return t, samples
