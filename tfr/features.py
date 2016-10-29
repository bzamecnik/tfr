"""
Example usage:

import matplotlib.pyplot as plt
import soundfile as sf
from analysis import read_frames

def analyze_mean_energy(file, frame_size=1024):
    frames, t, fs = read_frames(x, frame_size)
    y = mean_energy(frames)
    plt.semilogy(t, y)
    plt.ylim(0, 1)
"""

import numpy as np


def mean_power(x_frames):
    return np.sqrt(np.mean(x_frames**2, axis=-1))

def power(x_frames):
    return np.sqrt(np.sum(x_frames**2, axis=-1))

def mean_energy(x_frames):
    return np.mean(x_frames**2, axis=-1)

def energy(x_frames):
    return np.sum(x_frames**2, axis=-1)
