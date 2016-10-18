import subprocess

from synthesis import sample_time, fade, generate_and_save


def play(filename):
    subprocess.call(['afplay', filename])

def generate_and_play(func, filename='test.wav', **kwargs):
    t, samples = generate_and_save(func, filename, **kwargs)
    play(filename)
    return t, samples
