from __future__ import print_function, division

import numpy as np

class Tuning():
    """
    Equal temperament tuning - allows to convert between frequency and pitch.

    - unit pitch space
      - continous, unbounded
      - 1.0 ~ one octave
    - step pitch space
      - continous, unbounded
      - N steps ~ one octave
      - unit pitch space * N
    - unit pitch class space
      - continous, bounded [0, 1.0)
      - unit pitch space % 1.0
    - step pitch class space
      - continous, bounded [0, N)
      - unit step pitch space % N
    - integer step pitch space
      - discrete, unbounded
      - floor(step pitch space)
    - integer step pitch class space
      - discrete, bounded {0, 1, .. N - 1}
      - floor(step pitch class space)
    """
    def __init__(self, base_freq=440, steps_per_octave=12, octave_ratio=2):
        self.base_freq = base_freq
        self.steps_per_octave = steps_per_octave
        self.octave_ratio = octave_ratio

    def pitch_to_freq(self, pitch):
        factor = self.pitch_to_relative_freq(pitch)
        return factor * self.base_freq

    def freq_to_pitch(self, freq):
        rel_freq = freq / self.base_freq
        if self.octave_ratio == 2:
            p = np.log2(rel_freq)
        else:
            p = np.log(rel_freq) / np.log(2)
        return p * self.steps_per_octave

    def pitch_to_relative_freq(self, pitch):
        return pow(self.octave_ratio, pitch / self.steps_per_octave)

class PitchQuantizer():
    def __init__(self, tuning, bin_division=1):
        self.tuning = tuning
        self.bin_division = bin_division

    def quantize(self, freqs):
        """
        Quantizes frequencies to nearest pitch bins (with optional division of
        bins).
        """
        return np.round(self.tuning.freq_to_pitch(freqs) * self.bin_division) / self.bin_division
