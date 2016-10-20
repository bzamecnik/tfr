import numpy as np
import scipy.misc
import sys


def save_raw_spectrogram_bitmap(file_name, spectrogram):
    # input:
    # rows = frequency bins (low to high)
    # columns = time
    # output:
    # rows = frequency bins (bottom to top)
    # columns = time (left to right)
    scipy.misc.imsave(file_name, spectrogram.T[::-1])

def spectrogram_to_image(npz_file):
  save_raw_spectrogram_bitmap(npz_file + '.png', np.load(npz_file)['arr_0'])

if __name__ == '__main__':
    spectrogram_to_image(sys.argv[1])
