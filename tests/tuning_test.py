from tfr.tuning import pitch_to_relative_freq, pitch_to_freq


def test_pitch_to_relative_freq():
    assert 1. == pitch_to_relative_freq(0., steps_per_octave=1)
    assert 2. == pitch_to_relative_freq(1., steps_per_octave=1)
    assert 4. == pitch_to_relative_freq(2., steps_per_octave=1)
    assert 0.5 == pitch_to_relative_freq(-1., steps_per_octave=1)

    assert 1. == pitch_to_relative_freq(0., steps_per_octave=12)
    assert 2. == pitch_to_relative_freq(12., steps_per_octave=12)
    assert 4. == pitch_to_relative_freq(24., steps_per_octave=12)
    assert 0.5 == pitch_to_relative_freq(-12., steps_per_octave=12)

def test_pitch_to_freq():
    assert 440. == pitch_to_freq(0.)
    assert 880. == pitch_to_freq(12.)
    assert 1760. == pitch_to_freq(24.)
    assert 220. == pitch_to_freq(-12.)

    assert abs(466.1637615180899 - pitch_to_freq(1.)) < 1e-10
    assert abs(415.3046975799451 - pitch_to_freq(-1.)) < 1e-10
    assert abs(1318.5102276514797 - pitch_to_freq(12 + 7)) < 1e-10
