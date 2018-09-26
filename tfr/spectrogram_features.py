"""
The goal is to transform an audio signal into an STFT spectrogram in a form
suitable as features for machine learning.
"""

from __future__ import print_function, division

import numpy as np
import os

from .signal import SignalFrames
from .reassignment import reassigned_spectrogram, pitchgram


def spectrogram_features(file_name, frame_size, output_frame_size, hop_size, spectrogram_type, magnitudes='power_db_normalized'):
    signal_frames = SignalFrames(file_name, frame_size, hop_size, mono_mix=True)

    if spectrogram_type == 'stft':
        X = reassigned_spectrogram(signal_frames, output_frame_size,
            magnitudes=magnitudes, reassign_time=False, reassign_frequency=False)
    elif spectrogram_type == 'reassigned':
        X = reassigned_spectrogram(signal_frames, output_frame_size,
            magnitudes=magnitudes)
    elif spectrogram_type == 'pitchgram':
        X = pitchgram(signal_frames, output_frame_size, magnitudes=magnitudes)
    else:
        raise ValueError('unknown spectrogram type: %s' % spectrogram_type)

    return X

def spectrogram_features_to_file(input_filename, output_filename, frame_size, output_frame_size, hop_size, spectrogram_type, magnitudes='power_db'):
    X = spectrogram_features(input_filename, frame_size, output_frame_size, hop_size, spectrogram_type, magnitudes)
    np.savez_compressed(output_filename, X)
    # scipy.misc.imsave(output_filename.replace('.npz', '.png'), X.T[::-1])

def default_output_filename(input_file_name, type):
    return os.path.basename(input_file_name).replace('.wav', '_power_spectrogram_%s.npz' % type)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Extracts STFT magnitude spectrogram features.')
    parser.add_argument('input_file', metavar='INPUT', help='input file in WAV format')
    parser.add_argument('output_file', metavar='OUTPUT', nargs='?', help='output file in NumPy npz format')
    parser.add_argument('-b', '--frame-size', type=int, default=2048, help='STFT frame size')
    parser.add_argument('-p', '--hop-size', type=int, default=512, help='STFT hop size')
    parser.add_argument('-o', '--output-frame-size', type=int, default=512, help='output frame size')
    parser.add_argument('-t', '--type', default='stft', help='plain "stft", "reassigned" spectrogram or "pitchgram"')
    parser.add_argument('-m', '--magnitudes', default='power_db_normalized',
        choices=['linear', 'power', 'power_db', 'power_db_normalized'])

    return parser.parse_args()

def main():
    args = parse_args()

    output = args.output_file if args.output_file else default_output_filename(args.input_file, args.type)

    spectrogram_features_to_file(args.input_file, output, args.frame_size,
        args.output_frame_size, args.hop_size, args.type, args.magnitudes)

if __name__ == '__main__':
    main()
