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

def pitch_to_freq(pitch, base_freq=440, steps_per_octave=12, octave_ratio=2):
    factor = pitch_to_relative_freq(pitch, steps_per_octave, octave_ratio)
    return factor * base_freq

def pitch_to_relative_freq(pitch, steps_per_octave=1, octave_ratio=2):
    return pow(octave_ratio, pitch / steps_per_octave)


def experiment_errors_12tet_pitches_vs_harmonics():
    '''Generate pitches uniformly and examine how they are close to harmonics.'''
    def error(f):
        return f - round(f)
    indexes = list(range(0, 12 * 8 + 1))
    errors = [error(pitch_to_relative_freq(i, steps_per_octave=12)) \
        for i in indexes]
    plt.bar(indexes, errors)

if __name__ == '__main__'

    assert 1. == pitch_to_relative_freq(0., steps_per_octave=1)
    assert 2. == pitch_to_relative_freq(1., steps_per_octave=1)
    assert 4. == pitch_to_relative_freq(2., steps_per_octave=1)
    assert 0.5 == pitch_to_relative_freq(-1., steps_per_octave=1)

    assert 1. == pitch_to_relative_freq(0., steps_per_octave=12)
    assert 2. == pitch_to_relative_freq(12., steps_per_octave=12)
    assert 4. == pitch_to_relative_freq(24., steps_per_octave=12)
    assert 0.5 == pitch_to_relative_freq(-12., steps_per_octave=12)

    assert 440. == pitch_to_freq(0.)
    assert 880. == pitch_to_freq(12.)
    assert 1760. == pitch_to_freq(24.)
    assert 220. == pitch_to_freq(-12.)

    assert abs(466.1637615180899 - pitch_to_freq(1.)) < 1e-10
    assert abs(415.3046975799451 - pitch_to_freq(-1.)) < 1e-10
    assert abs(1318.5102276514797 - pitch_to_freq(12 + 7)) < 1e-10
