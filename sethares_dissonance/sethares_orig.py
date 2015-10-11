from __future__ import division
from numpy import exp, asarray, argsort, sort, sum, minimum
 
 
def dissmeasure(fvec, amp, model='min'):
    """
    Given a list of partials in fvec, with amplitudes in amp, this routine
    calculates the dissonance by summing the roughness of every sine pair
    based on a model of Plomp-Levelt's roughness curve.

    The older model (model='product') was based on the product of the two
    amplitudes, but the newer model (model='min') is based on the minimum
    of the two amplitudes, since this matches the beat frequency amplitude.
    """
    fvec = asarray(fvec)
    amp = asarray(amp)
 
    # used to stretch dissonance curve for different freqs:
    Dstar = 0.24  # point of maximum dissonance
    S1 = 0.0207
    S2 = 18.96
 
    C1 = 5
    C2 = -5
 
    # Plomp-Levelt roughness curve:
    A1 = -3.51
    A2 = -5.75
 
    ams = amp[argsort(fvec)]
    fvec = sort(fvec)
 
    D = 0
    for i in range(1, len(fvec)):
        Fmin = fvec[:-i]
        S = Dstar / (S1 * Fmin + S2)
        Fdif = fvec[i:] - fvec[:-i]
        if model == 'min':
            a = minimum(ams[:-i], ams[i:])
        elif model == 'product':
            a = ams[i:] * ams[:-i] # Older model
        else:
            raise ValueError('model should be "min" or "product"')
        Dnew = a * (C1 * exp(A1 * S * Fdif) + C2 * exp(A2 * S * Fdif))
        D += sum(Dnew)
 
    return D
 
 
if __name__ == '__main__':
    from numpy import array, linspace, append
    import matplotlib.pyplot as plt
 
    # Similar to Figure 3
    # http://sethares.engr.wisc.edu/consemi.html#anchor15619672
    # freq = 500 * array([1, 2, 3, 4, 5, 6])
    # amp = 0.88**array([0, 1, 2, 3, 4, 5])
    # alpharange = 2.3
    # method = 'min'
 
#    # Davide Verotta Figure 4 example
    freq = 261.63 * array([1, 2, 3, 4, 5, 6])
    amp = 1 / array([1, 2, 3, 4, 5, 6])
    alpharange = 2.0
    method = 'product'
 
    # call function dissmeasure for each interval
    diss = array([0])
    for alpha in linspace(1, alpharange, 1000):
        f = append(freq, alpha*freq)
        a = append(amp, amp)
        d = dissmeasure(f, a, method)
        diss = append(diss, d)
 
    import numpy as np
    np.savetxt('dissonances_verotta_orig.txt', diss)
    plt.plot(linspace(1, alpharange, len(diss)), diss)
    plt.xscale('log')
    plt.xlim(1, alpharange)
 
    plt.xlabel('frequency ratio')
    plt.ylabel('dissonance')
 
    for n, d in [(2,1), (3,2), (5,3), (4,3), (5,4), (6,5)]:
        plt.axvline(n/d, color='silver')
        plt.annotate(str(n) + ':' + str(d),
                     (n/d, max(diss)*2/3),
                     horizontalalignment='center')
 
    plt.show()
