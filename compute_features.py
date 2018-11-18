import os,sys
import numpy as np
import re
import scipy
import librosa


def read_audio(audio_path, target_fs=None, duration=4):
    (audio, fs) = librosa.load(audio_path, sr=target_fs, duration=duration)
    # print(fs)
    # if this is not a mono sounds file
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs, librosa.get_duration(filename=audio_path)


def readAudio(filein, duration=4):
    audioObj, sampleRate, duration = read_audio(filein, target_fs=16000, duration=duration)
    bitrate = audioObj.dtype

    try:
        maxv = np.finfo(bitrate).max
    except:
        maxv = np.iinfo(bitrate).max

    return audioObj.astype('float')/maxv, sampleRate, bitrate, duration


def readAudioScipy(filein):
    sampleRate, audioObj = scipy.io.wavfile.read(filein)
    bitrate = audioObj.dtype

    try:
        maxv = np.finfo(bitrate).max
    except:
        maxv = np.iinfo(bitrate).max
    return audioObj.astype('float'), sampleRate, bitrate


if __name__ == "__main__":
    db = '/home/ankur/Downloads/Others/mixed1'
    feature_path = '/home/ankur/Downloads/Others/features'
    n_window = 1024
    n_overlap = 256
    ham_win = np.hamming(n_window)

    for filename in os.listdir(db):
        if filename.endswith(".wav"):
            print(filename)
            audioObj, sampleRate, bitrate = readAudioScipy(os.path.join(db, filename))

            assert sampleRate == 16000,"Sample rate needs to be 16000"
            # print(audioObj.shape)
            audio = np.zeros((audioObj.shape[0],))
            clean = np.zeros((audioObj.shape[0],))
            noise = np.zeros((audioObj.shape[0],))

            audio = audioObj[:,0] + audioObj[:,1] #create mixture voice + accompaniment

            [f, t, mixed_spec] = scipy.signal.spectral.spectrogram(
                x=audio,
                window=ham_win,
                nperseg=n_window,
                noverlap=n_overlap,
                detrend=False,
                return_onesided=True,
                mode='magnitude')
            # print(mixed_spec.T.shape)

            clean = audioObj[:,0] #voice
            noise = audioObj[:,1] #accompaniment

            [f, t, clean_spec] = scipy.signal.spectral.spectrogram(
                x=clean,
                window=ham_win,
                nperseg=n_window,
                noverlap=n_overlap,
                detrend=False,
                return_onesided=True,
                mode='magnitude')
            # print(clean_spec.T.shape)

            [f, t, noise_spec] = scipy.signal.spectral.spectrogram(
                x=noise,
                window=ham_win,
                nperseg=n_window,
                noverlap=n_overlap,
                detrend=False,
                return_onesided=True,
                mode='magnitude')
            # print(noise_spec.T.shape)
            mask = np.zeros(mixed_spec.shape)

            for i in range(mixed_spec.shape[0]):
                for j in range(mixed_spec.shape[1]):
                    if clean_spec[i][j] >= noise_spec[i][j]:
                        mask[i][j] = 1
                    else:
                        mask[i][j] = 0

            # print(mask.shape)

            audioObj=None

            if not os.path.exists(feature_path):
                os.makedirs(feature_path)

            np.save(os.path.join(feature_path, filename.replace('.wav', '__spec.npy')), mixed_spec.T)
            np.save(os.path.join(feature_path, filename.replace('.wav', '__mask.npy')), mask.T)
