# - unit pitch space
#   - continous, unbounded
#   - 1.0 ~ one octave
# - step pitch space
#   - continous, unbounded
#   - N steps ~ one octave
#   - unit pitch space * N
# - unit pitch class space
#   - continous, bounded [0, 1.0)
#   - unit pitch space % 1.0
# - step pitch class space
#   - continous, bounded [0, N)
#   - unit step pitch space % N
# - integer step pitch space
#   - discrete, unbounded
#   - floor(step pitch space)
# - integer step pitch class space
#   - discrete, bounded {0, 1, .. N - 1}
#   - floor(step pitch class space)

# equal temperament

import numpy as np

def pitch_to_freq(pitch, base_freq=440, steps_per_octave=12, octave_ratio=2):
    factor = pitch_to_relative_freq(pitch, steps_per_octave, octave_ratio)
    return factor * base_freq

def pitch_to_relative_freq(pitch, steps_per_octave=1, octave_ratio=2):
    return pow(octave_ratio, pitch / steps_per_octave)

def freq_to_pitch(freq, base_freq=440, steps_per_octave=12, octave_ratio=2):
    rel_freq = freq / base_freq
    if octave_ratio == 2:
        p = np.log2(rel_freq)
    else:
        p = np.log(rel_freq) / np.log(2)
    return p * steps_per_octave

def pitch_bin_range(pitch_start=-4*12, pitch_end=5*12 + 8, pitch_step=1, base_freq=440):
    "generates a range of pitch bins and their frequencies"
    # [-48,67) -> [~27.5, 21096.2) Hz
    pitch_range = np.arange(pitch_start, pitch_end, pitch_step)
    bin_center_freqs = np.array([pitch_to_freq(f, base_freq=base_freq) for f in pitch_range])
    return pitch_range, bin_center_freqs

def quantize_freqs_to_pitch_bins(freqs, bin_division=1, freq_to_pitch=freq_to_pitch):
    "quantizes frequencies to nearest bins (with optional division of bins)"
    return np.round(freq_to_pitch(freqs) * bin_division) / bin_division
