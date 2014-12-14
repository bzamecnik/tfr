from __future__ import division
import numpy as np
  
def dissmeasure(fvec, amp, model='min'):
    """
    Given a list of partials in fvec, with amplitudes in amp, this routine
    calculates the dissonance by summing the roughness of every sine pair
    based on a model of Plomp-Levelt's roughness curve.

    The older model (model='product') was based on the product of the two
    amplitudes, but the newer model (model='min') is based on the minimum
    of the two amplitudes, since this matches the beat frequency amplitude.
    """
    fvec = np.copy(fvec)
    amp = np.copy(amp)
 
    # used to stretch dissonance curve for different freqs:
    Dstar = 0.24  # point of maximum dissonance
    S1 = 0.0207
    S2 = 18.96
 
    C1 = 5
    C2 = -5
 
    # Plomp-Levelt roughness curve:
    A1 = -3.51
    A2 = -5.75
 
    ams = amp[np.argsort(fvec)]
    fvec = np.sort(fvec)
 
    D = 0
    for i in range(1, len(fvec)):
        Fmin = fvec[:-i]
        S = Dstar / (S1 * Fmin + S2)
        Fdif = fvec[i:] - fvec[:-i]
        if model == 'min':
            a = np.minimum(ams[:-i], ams[i:])
        elif model == 'product':
            a = ams[i:] * ams[:-i] # Older model
        else:
            raise ValueError('model should be "min" or "product"')
        Dnew = a * (C1 * np.exp(A1 * S * Fdif) + C2 * np.exp(A2 * S * Fdif))
        D += np.sum(Dnew)
 
    return D
 
 
if __name__ == '__main__':
    import matplotlib.pyplot as plt
 
    # Similar to Figure 3
    # http://sethares.engr.wisc.edu/consemi.html#anchor15619672
    freqs = 500 * np.array([1, 2, 3, 4, 5, 6])
    amps = 0.88 ** np.array([0, 1, 2, 3, 4, 5])
    alpharange = 2.3
    method = 'min'
 
#    # Davide Verotta Figure 4 example
    # freqs = 261.63 * np.array([1, 2, 3, 4, 5, 6])
    # amps = 1 / np.array([1, 2, 3, 4, 5, 6])
    # alpharange = 2.0
    # method = 'product'

    # call function dissmeasure for each interval
    interval_count = 1000
    dissonances = np.array([])
    alphas = np.linspace(1, alpharange, interval_count)
    for alpha in alphas:
        # generate intervals by fixing one tone and varying the other
        other_freqs = alpha * freqs
        other_amps = amps
        # concat frequencies from both tones in the interval
        all_freqs = np.hstack([freqs, other_freqs])
        all_amps = np.hstack([amps, amps])
        d = dissmeasure(all_freqs, all_amps, method)
        dissonances = np.append(dissonances, d)

    plt.plot(alphas, dissonances)
    plt.xscale('log')
    plt.xlim(1, alpharange)
 
    plt.xlabel('frequency ratio')
    plt.ylabel('dissonance')
 
    for n, d in [(2,1), (3,2), (5,3), (4,3), (5,4), (6,5)]:
        plt.axvline(n/d, color='silver')
        plt.annotate(str(n) + ':' + str(d),
                     (n/d, max(dissonances)*2/3),
                     horizontalalignment='center')
 
    plt.show()
    # plt.savefig('plot.png')
