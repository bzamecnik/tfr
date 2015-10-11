from sklearn.cluster import KMeans
from scipy.signal import medfilt
from features import *
from sklearn.preprocessing import MinMaxScaler
from reassignment import *

#filename = '/Users/bzamecnik/Documents/harmoneye-labs/harmoneye/data/wav/c-scale-piano-mono.wav'

filename = '/Users/bzamecnik/Documents/harmoneye-labs/_inbox/02-Eleanor_Rigby.wav'

block_size = 2048
hop_size = 2048

song, fs = load_wav(filename)
x, times = split_to_blocks(song, block_size, hop_size=hop_size)

w = create_window(block_size)
X, X_cross_time, X_cross_freq, X_inst_freqs, X_group_delays = compute_spectra(x, w)

X_magnitudes = db_scale(abs(X) / X.shape[1])
X_reassigned = requantize_spectrogram(X_cross_time, X_inst_freqs)
X_reassigned_real = real_half(X_reassigned)

X_power = mean_power(X_magnitudes)

kmeans = KMeans(n_clusters=2)
 
#kmeans = KMeans(n_clusters=13)
kmeans.fit(X_reassigned_real)
X_kmeans = kmeans.transform(X_reassigned_real)

classes = medfilt(X_kmeans.argmin(axis=1), 7)

plot(X_power)
plot(classes)
