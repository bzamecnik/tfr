import numpy as np

from tfr.analysis import split_to_blocks

def test_split_to_blocks():
    assert np.array([
        [ 0,  1,  2,  3,  4,  5,  6,  7],
        [ 6,  7,  8,  9, 10, 11, 12, 13],
        [12, 13, 14, 15, 16, 17, 18, 19],
        [18, 19, 20, 21, 22,  0,  0,  0],
    ]) == split_to_blocks(np.arange(23), 8, 6)
