from __future__ import division
import numpy as np

def dissonance(freqs, amps, model='min'):
    """
    Given a list of partials in freqs, with amplitudes in amp, this routine
    calculates the dissonance by summing the roughness of every sine pair
    based on a model of Plomp-Levelt's roughness curve.

    The older model (model='product') was based on the product of the two
    amplitudes, but the newer model (model='min') is based on the minimum
    of the two amplitudes, since this matches the beat frequency amplitude.
    """
    
    if model == 'min':
        reduce_amplitudes = np.minimum
    elif model == 'product':
        # Older model
        reduce_amplitudes = lambda x, y: x * y
    else:
        raise ValueError('model should be "min" or "product"')
    
    freqs = np.copy(freqs)
    amps = np.copy(amps)

    # used to stretch dissonance curve for different freqs:
    Dstar = 0.24  # point of maximum dissonance
    S1 = 0.0207
    S2 = 18.96

    C1 = 5
    C2 = -5

    # Plomp-Levelt roughness curve:
    A1 = -3.51
    A2 = -5.75

    sorted_freq_indices = np.argsort(freqs)
    amps = amps[sorted_freq_indices]
    freqs = freqs[sorted_freq_indices]

    dissonance = 0
    for i in range(1, len(freqs)):
        Fmin = freqs[:-i]
        S = Dstar / (S1 * Fmin + S2)
        freqs_diff = freqs[i:] - freqs[:-i]
        a = reduce_amplitudes(amps[:-i], amps[i:])
        partial_dissonance = np.sum(
            a * (C1 * np.exp(A1 * S * freqs_diff) +
                 C2 * np.exp(A2 * S * freqs_diff)))
        dissonance += partial_dissonance

    return dissonance

def sethares_params():
    # Similar to Figure 3
    # http://sethares.engr.wisc.edu/consemi.html#anchor15619672
    base_freq = 500
    freqs = base_freq * np.array([1, 2, 3, 4, 5, 6])
    amps = 0.88 ** np.array([0, 1, 2, 3, 4, 5])
    alpharange = 2.3
    method = 'min'
    return freqs, amps, alpharange, method

def verotta_params():
    # Davide Verotta - Figure 4 example
    base_freq = 261.63
    freqs = base_freq * np.array([1, 2, 3, 4, 5, 6])
    amps = 1 / np.array([1, 2, 3, 4, 5, 6])
    alpharange = 2.0
    method = 'product'
    return freqs, amps, alpharange, method

def evaluate(freqs, amps, alpharange, method, interval_count=1000):
    '''
    Evaluate the dissonance function for each interval.
    Generate intervals by fixing one tone and varying the other.
    '''
    dissonances = np.zeros(interval_count)
    alphas = np.linspace(1, alpharange, interval_count)
    all_amps = np.hstack([amps, amps])
    for i, alpha in enumerate(alphas):
        other_freqs = alpha * freqs
        # concat frequencies from both tones in the interval
        all_freqs = np.hstack([freqs, other_freqs])
        dissonances[i] = dissonance(all_freqs, all_amps, method)
    return dissonances, alphas

def plot_dissonance_curve(dissonances, alphas):
    plt.plot(alphas, dissonances)
    plt.xscale('log')
    plt.xlim(min(alphas), max(alphas))

    plt.xlabel('frequency ratio')
    plt.ylabel('dissonance')

    # annotate the simple ratios (just intervals in 12-TET)
    # with vertical lines and labels
    for n, d in [(2,1), (3,2), (5,3), (4,3), (5,4), (6,5)]:
        x = n / d
        # position the label to some free space
        y = max(dissonances) * 2 / 3
        plt.axvline(x, color='silver')
        plt.annotate('%d:%d' % (n, d), (x, y),
                horizontalalignment='center')

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    params = sethares_params()
    # params = verotta_params()
    dissonances, alphas = evaluate(*params)

    plot_dissonance_curve(dissonances, alphas)
    plt.show()    
    # plt.savefig('plot.png')
