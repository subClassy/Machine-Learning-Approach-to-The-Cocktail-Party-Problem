import os
from pydub import AudioSegment
import numpy as np
import re
from scipy import signal, io
import librosa

noise_folder = '/home/ankur/Downloads/Others/noise'
clean_folder = '/home/ankur/Downloads/Others/dev-clean'
mixed_folder = '/home/ankur/Downloads/Others/mixed'

noise_files = os.listdir(noise_folder)
clean_files = os.listdir(clean_folder)


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


def create_mixed_audio(zipped_list):
    for (noise, clean) in list(zipped_list):
        if clean.split('.')[-1] != 'txt':
            noise_path = noise_folder + '/' + noise
            clean_path = clean_folder + '/' + clean

            # sound1 = AudioSegment.from_file(clean_path)
            # sound2 = AudioSegment.from_file(noise_path)

            # combined = sound1.overlay(sound2)

            clean_obj, sample_rate, duration = read_audio(clean_path, target_fs=16000)
            noise_obj, sample_rate, noise_duration = read_audio(noise_path, target_fs=16000, duration=duration)

            audio = np.zeros((clean_obj.shape[0], 2))
            to_pad = clean_obj.shape[0] - noise_obj.shape[0]

            if to_pad > 0:
                noise_obj = np.pad(noise_obj, (0, to_pad), 'constant')
            elif to_pad < 0:
                noise_obj = noise_obj[:clean_obj.shape[0]]

            audio[:, 0] = clean_obj
            audio[:, 1] = noise_obj

            filename = clean.split('.')[0] + '__' + noise.split('.')[0] + '.wav'
            file_path = mixed_folder + '/' + filename
            print(filename)
            maxn = np.iinfo("int16").max
            print(maxn)

            io.wavfile.write(filename=file_path, rate=16000, data=(audio * maxn).astype("int16"))


batches = len(clean_files) // len(noise_files)
size = len(noise_files)
size_o = len(clean_files)
zipped = []
current_idx = 0

for clean_file in clean_files:
    for _ in range(3):
        zipped.append((noise_files[current_idx], clean_file))
        current_idx += 1

        if current_idx == size:
            current_idx = 0

# print(zipped)
create_mixed_audio(zipped)

# for i in range(batches):
#     zipped = zip(noise_files, clean_files[i * size: (i + 1) * size])
#     create_mixed_audio(zipped)

# if len(clean_files) % size > 0:
#     zipped = zip(noise_files, clean_files[batches * size:])
#     create_mixed_audio(zipped)