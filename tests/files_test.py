import numpy as np
import os
import tempfile

# Note that nose prepends the project directory to the sys.path so that we
# do not import from the system-wide package!

from tfr import files


def test_array_should_be_saved_to_a_file():
    fs = 1000
    x = sine_wave(5.0, fs)
    with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
        file_name = tmpfile.name
        files.save_wav(x, file_name, fs=fs)
        assert os.path.exists(file_name)

def test_array_should_be_loaded_from_a_file():
    fs = 1000
    x = sine_wave(5.0, fs)
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        file_name = tmpfile.name
        files.save_wav(x, file_name, fs=fs)
        x_loaded, fs_loaded = files.load_wav(file_name)

        assert fs_loaded == fs
        # 16-bit quantization error is OK
        assert np.allclose(x_loaded, x, atol=(1 / (((2**15)) - 1)))

def sine_wave(duration, fs):
    return np.sin(np.linspace(0, duration, duration * fs))
