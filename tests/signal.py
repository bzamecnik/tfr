from __future__ import print_function, division

import numpy as np

from tfr import SignalFrames

def test_split_to_frames():
    signal_frames = SignalFrames(np.arange(23), frame_size=8, hop_size=6,
        sample_rate=44100)
    assert np.allclose(np.array([
        [ 0,  1,  2,  3,  4,  5,  6,  7],
        [ 6,  7,  8,  9, 10, 11, 12, 13],
        [12, 13, 14, 15, 16, 17, 18, 19],
        [18, 19, 20, 21, 22,  0,  0,  0],
    ]), signal_frames.frames)
