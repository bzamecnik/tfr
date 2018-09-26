from __future__ import print_function, division

from tfr import Tuning


def test_pitch_to_relative_freq():
    tuning_step1 = Tuning(steps_per_octave=1)
    tuning_step12 = Tuning(steps_per_octave=12)
    assert 1. == tuning_step1.pitch_to_relative_freq(0.)
    assert 2. == tuning_step1.pitch_to_relative_freq(1.)
    assert 4. == tuning_step1.pitch_to_relative_freq(2.)
    assert 0.5 == tuning_step1.pitch_to_relative_freq(-1.)

    assert 1. == tuning_step12.pitch_to_relative_freq(0.)
    assert 2. == tuning_step12.pitch_to_relative_freq(12.)
    assert 4. == tuning_step12.pitch_to_relative_freq(24.)
    assert 0.5 == tuning_step12.pitch_to_relative_freq(-12.)

def test_pitch_to_freq():
    tuning = Tuning()
    assert 440. == tuning.pitch_to_freq(0.)
    assert 880. == tuning.pitch_to_freq(12.)
    assert 1760. == tuning.pitch_to_freq(24.)
    assert 220. == tuning.pitch_to_freq(-12.)

    assert abs(466.1637615180899 - tuning.pitch_to_freq(1.)) < 1e-10
    assert abs(415.3046975799451 - tuning.pitch_to_freq(-1.)) < 1e-10
    assert abs(1318.5102276514797 - tuning.pitch_to_freq(12 + 7)) < 1e-10

# TODO: test:
# - freq_to_pitch()
# - PitchQuantizer
# - various configurations of Tuning
