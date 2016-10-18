import scipy


def save_raw_spectrogram_bitmap(file_name, spectrogram):
    # input:
    # rows = frequency bins (low to high)
    # columns = time
    # output:
    # rows = frequency bins (bottom to top)
    # columns = time (left to right)
    scipy.misc.imsave(file_name, spectrogram.T[::-1])
