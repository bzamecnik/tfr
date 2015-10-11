"""
The goal is to transform an audio signal into an STFT spectrogram in a form
suitable as features for machine learning.
"""

import numpy as np
import os

from spectrogram import create_window, magnitude_spectrum
from files import load_wav
from analysis import split_to_blocks

def spectrogram_features(input_filename, output_filename, block_size, hop_size, to_log=True):
    song, fs = load_wav(input_filename)
    x, times = split_to_blocks(song, block_size, hop_size=hop_size)
    w = create_window(block_size)
    X = magnitude_spectrum(x * w)
    if to_log:
        # dbFS
        X = 20 * np.log10(np.maximum(1e-6, X))
    np.savez_compressed(output_filename, X)

def default_output_filename(input_file_name):
    return os.path.basename(input_file_name).replace('.wav', '_spectrogram.npz')

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Extracts SFTF magnitude spectrogram features.')
    parser.add_argument('input_file', metavar='INPUT', help='input file in WAV format')
    parser.add_argument('--output', help='output file in NumPy npz format')
    parser.add_argument('-b', '--block-size', type=int, default=2048, help='STFT block size')
    parser.add_argument('-p', '--hop-size', type=int, default=512, help='STFT hop size')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    output = args.output if args.output else default_output_filename(args.input_file)

    spectrogram_features(args.input_file, output, args.block_size, args.hop_size)
