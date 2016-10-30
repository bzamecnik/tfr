for spec_type in stft reassigned pitchgram; do
  python -m tfr.spectrogram_features \
    she_brings_to_me.wav she_brings_to_me_$spec_type.npz \
    -t $spec_type -b 4096 -p 2048 --output-frame-size=2048
done
