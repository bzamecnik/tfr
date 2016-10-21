import numpy as np
import os
import soundfile as sf
import tempfile

# Note that nose prepends the project directory to the sys.path so that we
# do not import from the system-wide package!

import tfr.files

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

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


def test_reading_is_same_soundfile_wavfile():
    audio_file = os.path.join(DATA_DIR, 'she_brings_to_me.wav')
    x_sf, fs_sf = sf.read(audio_file)
    x_wf, fs_wf = tfr.files.load_wav(audio_file, mono_mix=False)
    assert fs_sf == fs_wf
    quantization_error = 2 ** (-15)
    print('read soundfile/wavfile: max abs error', abs(x_sf - x_wf).max())
    assert np.allclose(x_sf, x_wf, atol=quantization_error)


def sine_wave(duration, fs):
    return np.sin(np.linspace(0, duration, duration * fs))
