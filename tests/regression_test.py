"""
A regression test for computing three kinds of spectrogram.

Just to ensure we didn't break anything.
"""

import numpy as np
import os
import soundfile as sf
from tfr.spectrogram_features import spectrogram_features

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def test_spectrograms():
    for spectrogram_type in ['stft', 'reassigned', 'chromagram']:
        yield assert_spectrogram_is_ok, spectrogram_type

def assert_spectrogram_is_ok(spectrogram_type):
    audio_file = os.path.join(DATA_DIR, 'she_brings_to_me.wav')
    X = spectrogram_features(audio_file, frame_size=4096, hop_size=2048,
        output_frame_size=2048,
        spectrogram_type=spectrogram_type, to_log=True)
    npz_file = os.path.join(DATA_DIR, 'she_brings_to_me_%s.npz' % spectrogram_type)
    X_expected = np.load(npz_file)['arr_0']
    assert X.shape == X_expected.shape, (X.shape, X_expected.shape)
    print('spectrogram [%s]: max abs error' % spectrogram_type, abs(X - X_expected).max())
    assert np.allclose(X, X_expected)
