import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from synthesis import *
from reassignment import arg, compute_spectra
from spectrogram import split_to_blocks, create_window

t = sample_time(0, 1)

f_fund = 440
f_partial = 2 * f_fund
x_fund = sine(t, f_fund)
#x_coherent = x_fund + sine(t, f_partial)
#x_incoherent = x_fund + sine(t, f_partial, phase=0.4*np.pi)

block_size, hop_size = 2048, 512
w = create_window(block_size)

steps = 25
signals = [x_fund + sine(t, f_partial, phase=phi*2*np.pi) for phi in np.linspace(0, 1, steps)]

def phase_spectrum(x):
    return arg(compute_spectra(split_to_blocks(x, block_size, hop_size)[0], w)[0])

phase_spectra = [phase_spectrum(x) for x in signals]

# for i, ps in enumerate(phase_spectra):
#     imshow(ps[:20,15:45].T, interpolation='nearest', origin='lower')
#     savefig('phase_spectrum_%d.png' % i)

fig = plt.figure()

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i, ps in enumerate(phase_spectra):
    im = plt.imshow(ps[:20,15:45].T,
        interpolation='nearest', origin='lower', cmap=plt.cm.hsv)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False, repeat_delay=1000)

#ani.save('dynamic_images.mp4')

plt.show()
