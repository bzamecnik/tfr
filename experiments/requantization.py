def requantize2d(X_group_delays, X_inst_freqs, times, fs, weights=None):
    block_duration = block_size / fs
    block_center_time = block_duration / 2
    X_time = np.tile(times + block_center_time, (X_group_delays.shape[1], 1)).T \
        + X_group_delays
    time_range = (times[0], times[-1] + block_duration)
    freq_range = (0, 1)
    bins = X_inst_freqs.shape
    counts, x_edges, y_edges = np.histogram2d(
        X_time.flatten(), X_inst_freqs.flatten(),
        weights=weights.flatten(),
        range=(time_range, freq_range),
        bins=bins)
    return counts, x_edges, y_edges
