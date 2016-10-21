# tfr - time-frequency reassignment in Python

[![PyPI version](https://img.shields.io/pypi/v/tfr.svg)](https://pypi.python.org/pypi/tfr)
![Supported Python versions](https://img.shields.io/pypi/pyversions/tfr.svg)
![License](https://img.shields.io/pypi/l/tfr.svg)

Spectral audio feature extraction using [time-frequency reassignment](https://en.wikipedia.org/wiki/Reassignment_method).

![reassigned spectrogram illustration](reassigned-spectrogram.png)

Besides normals spectrograms it allows to compute reassigned spectrograms, transform them (eg. to log-frequency scale) and requantize them (eg. to musical pitch bins). This is useful to obtain good features for audio analysis or machine learning on audio data.

A reassigned spectrogram often provides more precise localization of energy in the time-frequency plane than a plain spectrogram. Roughly said in the reassignment method we use the phase (which is normally discarded) and move the samples on the time-frequency plane to a more suitable place computed from derivatives of the phase.

## Installation

```
pip install tfr
```

Or for development (all code changes will be available):

```
git clone https://github.com/bzamecnik/tfr.git
pip install -e tfr
```

## Usage

### Extract a chromagram from an audio file

```
from tfr.analysis import split_to_blocks, to_mono
from tfr.reassignment import chromagram
from tfr.spectrogram import create_window
import soundfile as sf

x, fs = sf.read('audio.flac')
block_size = 4096
window = create_window(block_size)
x_blocks, x_times = split_to_blocks(to_mono(x), block_size=block_size, hop_size=2048, fs=fs)

# input:
#   - blocks of mono audio signal normalized to [0.0, 1.0]
#   - shape: (block_count, block_size)
#   - bin_range is in pitch bins where 0 = 440 Hz (A4)
# output:
#   - chromagram of shape (block_count, bin_count)
#   - values are log-magnitudes in dBFS [-120.0, bin_count]
x_chromagram = chromagram(x_blocks, window, fs=fs, to_log=True, bin_range=[-48, 67], bin_division=1)
```

### Extract features via CLI

```
# basic STFT spectrogram
python -m tfr.spectrogram_features audio.flac spectrogram.npz
# reassigned STFT spectrogram
python -m tfr.spectrogram_features audio.flac -t reassigned reassigned_spectrogram.npz
# reassigned chromagram
python -m tfr.spectrogram_features audio.flac -t chromagram chromagram.npz
```

Look for other options:

```
python -m tfr.spectrogram_features --help
```

### scikit-learn transformer

In order to extract chromagram features within a sklearn pipeline, we can use `ChromagramTransformer`:

```
import soundfile as sf
x, fs = sf.read('audio.flac')

from tfr.analysis import to_mono
from tfr.preprocessing import ChromagramTransformer
ct = ChromagramTransformer(sample_rate=fs)
x_chromagram = ct.transform(to_mono(x))

# output:
#  - shape: (block_count, bin_count)
#   - values in dBFB normalized to [0.0, 1.0]
```

## Status

Currently it's alpha. I'm happy to extract it from some other project into a separate repo and package it. However, the API must be completely redone to be more practical and obvious.

## About

- Author: Bohumír Zámečník ([@bzamecnik](http://twitter.com/bzamecnik))
- License: MIT
