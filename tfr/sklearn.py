from __future__ import absolute_import

from sklearn.base import BaseEstimator, TransformerMixin

from .signal import SignalFrames
from .reassignment import pitchgram


class PitchgramTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sample_rate=44100, frame_size=4096, hop_size=2048,
        output_frame_size=None,
        bin_range=[-48, 67], bin_division=1):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        # if no output frame size is specified the input hop size is the default
        self.output_frame_size = output_frame_size if output_frame_size is not None else hop_size
        self.bin_range = bin_range
        self.bin_division = bin_division

    def transform(self, X, **transform_params):
        """
        Transforms audio clip X into a normalized pitchgram.
        Input: X - mono audio clip - numpy array of shape (samples,)
        Output: X_pitchgram - numpy array of shape (frames, bins)
        """
        signal_frames = SignalFrames(X, self.frame_size, self.hop_size,
            self.sample_rate, mono_mix=True)
        X_pitchgram = pitchgram(
            signal_frames,
            self.output_frame_size,
            magnitudes='power_db_normalized',
            bin_range=self.bin_range,
            bin_division=self.bin_division)
        return X_pitchgram

    def fit(self, X, y=None, **fit_params):
        return self
