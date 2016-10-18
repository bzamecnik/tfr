"""
The goal is to transform an audio signal into an STFT spectrogram in a form
suitable as features for machine learning.
"""

import numpy as np
import os

from .spectrogram import create_window, magnitude_spectrum
from .files import load_wav
from .analysis import split_to_blocks
from .reassignment import reassigned_spectrogram, chromagram

def stft_spectrogram(x, w, to_log):
    X = magnitude_spectrum(x * w) ** 2
    if to_log:
        # dbFS
        X = 20 * np.log10(np.maximum(1e-6, X))
    return X

def spectrogram_features(song, fs, block_size, hop_size, spectrogram_type, to_log=True):
    x, times = split_to_blocks(song, block_size, hop_size=hop_size)
    w = create_window(block_size)

    if spectrogram_type == 'stft':
        spectrogram_func = stft_spectrogram
    elif spectrogram_type == 'reassigned':
        spectrogram_func = reassigned_spectrogram
    elif spectrogram_type == 'chromagram':
        spectrogram_func = lambda x, w, to_log: chromagram(x, w, fs, to_log=to_log)

    X = spectrogram_func(x, w, to_log)
    return X

def spectrogram_features_to_file(input_filename, output_filename, block_size, hop_size, spectrogram_type, to_log=True):
    song, fs = load_wav(input_filename)
    X = spectrogram_features(song, fs, block_size, hop_size, spectrogram_type, to_log)
    np.savez_compressed(output_filename, X)
    # scipy.misc.imsave(output_filename.replace('.npz', '.png'), X.T[::-1])

def default_output_filename(input_file_name, type):
    return os.path.basename(input_file_name).replace('.wav', '_power_spectrogram_%s.npz' % type)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Extracts STFT magnitude spectrogram features.')
    parser.add_argument('input_file', metavar='INPUT', help='input file in WAV format')
    parser.add_argument('output_file', metavar='OUTPUT', nargs='?', help='output file in NumPy npz format')
    parser.add_argument('-b', '--block-size', type=int, default=2048, help='STFT block size')
    parser.add_argument('-p', '--hop-size', type=int, default=512, help='STFT hop size')
    parser.add_argument('-t', '--type', default='stft', help='plain "stft", "reassigned" spectrogram or "chromagram"')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    output = args.output_file if args.output_file else default_output_filename(args.input_file, args.type)

    spectrogram_features_to_file(args.input_file, output, args.block_size, args.hop_size, args.type)
