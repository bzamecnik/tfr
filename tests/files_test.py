import numpy as np
import os
import soundfile as sf
import tempfile

# Note that nose prepends the project directory to the sys.path so that we
# do not import from the system-wide package!

import tfr.files


def test_array_should_be_saved_to_a_file():
    fs = 1000
    x = sine_wave(5.0, fs)
    with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
        file_name = tmpfile.name
        tfr.files.save_wav(x, file_name, fs=fs)
        assert os.path.exists(file_name)


def test_read_write_wavfile():
    """
    Writing and reading an array should be within quantization error.
    """
    fs = 44100
    x = 0.5 * sine_wave(1.0, fs)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
        tfr.files.save_wav(x, tmpfile.name, fs)
        y, _ = tfr.files.load_wav(tmpfile.name)
    quantization_error = 2 ** (-15)
    print('[wavfile] max absolute error:', abs(y-x).max())
    assert np.allclose(x, y, atol=quantization_error)


def test_read_write_soundfile():
    """
    Writing and reading an array should be within quantization error.
    """
    fs = 44100
    x = 0.5 * sine_wave(1.0, fs)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
        sf.write(tmpfile.name, x, fs)
        y, _ = sf.read(tmpfile.name)
    quantization_error = 2 ** (-15)
    print('[soundfile] max absolute error:', abs(y-x).max())
    assert np.allclose(x, y, atol=quantization_error)


def sine_wave(duration, fs):
    return np.sin(np.linspace(0, duration, duration * fs))
