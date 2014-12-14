import numpy as np
import matplotlib.pyplot as plt

def plomp_levelt_roughness(x):
    return np.exp(-3.5 * x) - np.exp(-5.75 * x)

def normal_roughness(x):
    return np.exp(-0.5 * x**2)

if __name__ = '__main__':
    x = np.logspace(-4, 2, 1000)
    
    y_pl = plomp_levelt_roughness(x)
    amplitude = max(y_pl)
    print(amplitude)
    center = np.log(x[np.argmax(y_pl)])
    print(center)
    y_lognormal = amplitude * normal_roughness(np.log(x) - center)
    
    # fig = plt.figure()
    # plt.xlim(0, 3)
    # plt.plot(x, y_pl, label='Plomp-Levelt')
    # plt.plot(x, y_lognormal, label='log-normal')
    # plt.legend()
    # plt.savefig('roughness-plomp-levelt+log-normal-linear.png')

    fig = plt.figure()
    plt.plot(x, y_pl, label='Plomp-Levelt')
    plt.plot(x, y_lognormal, label='log-normal')
    plt.xscale('log')
    plt.axvline(x[argmax(y_pl)], color='silver')
    plt.legend()
    # plt.savefig('roughness-plomp-levelt+log-normal-log.png')
