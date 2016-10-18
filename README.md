# tfr

Spectral audio feature extraction using [time-frequency reassignment](en.wikipedia.org/wiki/Reassignment_method).

![reassigned spectrogram illustration](reassigned-spectrogram.png)

Besides normals spectrograms it allows to compute reassigned spectrograms, transform them (eg. to log-frequency scale) and requantize them (eg. to musical pitch bins). This is useful to obtain good features for audio analysis or machine learning on audio data.

A reassigned spectrogram often provides more precise localization of energy in the time-frequency plane than a plain spectrogram. Roughly said in the reassignment method we use the phase (which is normally discarded) and move the samples on the time-frequency plane to a more suitable place computed from derivatives of the phase.
