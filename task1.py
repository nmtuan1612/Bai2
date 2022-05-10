import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np

audio_file = "audio/c1.wav"

# load audio files with librosa
signal, sr = librosa.load(audio_file)

len(signal), sr

# more parameters from librosa.feature.melspectrogram
mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
print("mfccs-shape:", mfccs.shape)

plt.figure(figsize=(25, 10))
librosa.display.specshow(mfccs,
                         x_axis="time",
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.show()

delta_mfccs = librosa.feature.delta(mfccs)
print("delta_mfccs-shape:", delta_mfccs.shape)

delta2_mfccs = librosa.feature.delta(mfccs, order=2)
print("delta2_mfccs-shape:", delta2_mfccs.shape)

plt.figure(figsize=(25, 10))
librosa.display.specshow(delta_mfccs,
                         x_axis="time",
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.show()

plt.figure(figsize=(25, 10))
librosa.display.specshow(delta2_mfccs,
                         x_axis="time",
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.show()

mfccs_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
print("mfccs_features-shape:", mfccs_features.shape)