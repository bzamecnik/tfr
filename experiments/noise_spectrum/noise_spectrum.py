import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

def dft(x):
    N = len(x)
    e = 1j * 2 * np.pi * np.arange(N) / N
    X = np.array([np.dot(x, np.exp(k * e)) for k in range(N)])
    return X

def time_samples(duration, fs):
    return np.arange(duration * fs) / fs

def sine(t, f):
    return np.sin(f * 2 * pi * t)

def noise(t):
    return np.random.random(len(t))

if __name__ == '__main__':
    fs = 1000
    t = time_samples(1, fs)
    
    x_sine = sine(t, 1)
    X_sine = dft(x_sine)
    
    x_noise = noise(t)
    X_noise = dft(x_noise)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    ax1.plot(t, x_sine, label=('sine 1 Hz'))
    ax2.stem(t, abs(X_sine) / (2*len(X_sine)), label='sine spectrum')
    ax3.plot(t, x_noise, label='white noise')
    ax4.stem(t, abs(X_noise) / (2*len(X_noise)), label='noise spectrum')
    plt.show()
