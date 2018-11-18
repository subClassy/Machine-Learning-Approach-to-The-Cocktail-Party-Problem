import math
import numpy as np
import pyaudio
import sys
import os
import wave
import matplotlib.pyplot as plt
# third-party sounds processing and visualization library
import librosa
import librosa.display
# signal processing library
from scipy import signal
import pickle

freq1 = 512.
freq2 = 1024.
n_window = 1024
# overlap between adjacent FFT windows
n_overlap = 256
# number of mel frequency bands to generate
n_mels = 64

# fmin and fmax for librosa filters in Hz - used for visualization purposes only
fmax = max(freq1, freq2) + 1000.
fmin = 0.
fontsize = 14

mm = '174-50561-0010__door_wood_knock_1-52290-A-30.wav'
audio_path = '/home/ankur/Downloads/Others/mixed/' + mm
sample_rate = 16000
# audio is a 1D time series of the sound
# can also use (audio, fs) = soundfile.read(audio_path)
(audio, fs) = librosa.load(audio_path, sr = sample_rate, duration = 4)
# print(fs)
# audio = librosa.resample(audio, orig_sr=fs, target_sr=sample_rate)
print(audio.shape)
# check that native bitrate matches our assumed sample rate
# assert(int(fs) == int(sample_rate))
# Make a new figure
# sample_rate = fs
plt.figure(figsize=(18, 16), dpi= 60, facecolor='w', edgecolor='k')
# plt.subplot(211)
# # Display the spectrogram on a mel scale
# librosa.display.waveplot(audio, int(sample_rate), max_sr = int(sample_rate))
# plt.title('Raw audio waveform @ %d Hz' % sample_rate, fontsize = fontsize)
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")

# plt.subplot(212)
melW =librosa.filters.mel(sr=sample_rate, n_fft=n_window, n_mels=n_mels, fmin=fmin, fmax=fmax)

ham_win = np.hamming(n_window)
[f, t, x] = signal.spectral.spectrogram(
    x=audio,
    window=ham_win,
    nperseg=n_window,
    noverlap=n_overlap,
    detrend=False,
    return_onesided=True,
    mode='magnitude')

np.save('mixed', x)
# print(melW.T.shape)
# print(melW.T)
# x = np.dot(x.T, melW.T)
# x = np.log(x + 1e-8)
# x = x.astype(np.float32)
print(x.T.shape)
print(x.T)
print(f.shape)
print(t.shape)

librosa.display.specshow(x.T, sr=sample_rate, x_axis='time', y_axis='hz', x_coords=np.linspace(0, 1, x.shape[0]))
plt.xlabel("Time (s)")
plt.title("Magnitude Spectrogram of Clean Audio", fontsize = fontsize)
# optional colorbar plot
# plt.colorbar(format='%+02.0f dB')
plt.show()