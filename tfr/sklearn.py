from sklearn.base import BaseEstimator, TransformerMixin

from .analysis import SignalFrames
from .spectrogram import create_window
from .reassignment import chromagram


class ChromagramTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sample_rate=44100, frame_size=4096, hop_size=2048,
        bin_range=[-48, 67], bin_division=1):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        # TODO: make this configurable
        self.output_frame_size = hop_size
        self.bin_range = bin_range
        self.bin_division = bin_division

    def transform(self, X, **transform_params):
        """
        Transforms audio clip X into a normalized chromagram.
        Input: X - mono audio clip - numpy array of shape (samples,)
        Output: X_chromagram - numpy array of shape (frames, bins)
        """
        signal_frames = SignalFrames(X, self.frame_size, self.hop_size,
            self.sample_rate, mono_mix=True)
        X_chromagram = chromagram(
            signal_frames,
            create_window,
            self.output_frame_size,
            to_log=True,
            bin_range=self.bin_range,
            bin_division=self.bin_division)
        # map from raw dB [-120.0, 0] to [0.0, 1.0]
        X_chromagram = (X_chromagram / 120) + 1
        return X_chromagram

    def fit(self, X, y=None, **fit_params):
        return self
