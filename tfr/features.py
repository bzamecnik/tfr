"""
Example usage:

import matplotlib.pyplot as plt
import soundfile as sf
from analysis import read_blocks

def analyze_mean_energy(file, block_size=1024):
    blocks, t, fs = read_blocks(x, block_size)
    y = mean_energy(blocks)
    plt.semilogy(t, y)
    plt.ylim(0, 1)
"""

import numpy as np


def mean_power(x_blocks):
    return np.sqrt(np.mean(x_blocks**2, axis=-1))

def power(x_blocks):
    return np.sqrt(np.sum(x_blocks**2, axis=-1))

def mean_energy(x_blocks):
    return np.mean(x_blocks**2, axis=-1)

def energy(x_blocks):
    return np.sum(x_blocks**2, axis=-1)
