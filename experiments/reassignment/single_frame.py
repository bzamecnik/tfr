from scipy.signal import chirp

from analysis import split_to_blocks
from synthesis import generate_and_save
from reassignment import compute_spectra

block_size = 2048
fs = 44100
f1, f2 = 440, 880
duration = block_size / fs
times, x = generate_and_save(lambda t: chirp(t, f1, duration, f2),
    duration=duration, fade_ends=False)

x_blocks = split_to_blocks(x, block_size, block_size)
X, X_cross_time, X_cross_freq, X_inst_freqs, X_group_delays = compute_spectra(x_blocks[0], w)


idx = (X_inst_freqs >= f1 / fs) & (X_inst_freqs <= f2 / fs)
plt.scatter(X_group_delays[idx], X_inst_freqs[idx], alpha=0.5, s=abs(X)[idx], c=abs(X)[idx])
