import numpy as np
import os
from keras.utils import Sequence
from scipy.io import wavfile
from scipy.signal import spectral
import matplotlib.pyplot as plt
from librosa.display import specshow
import librosa


class Dataset(Sequence):
    def __init__(self, feature_path):
        self.feature_path = feature_path
        self.files = list(filter(lambda filename: filename.endswith('__spec.npy'), os.listdir(feature_path)))
        self.current_file_index = 0
        self.num_samples = 20
        self.dataset_len = self.get_dataset_len()
        self.batch_size = 15

        self.load_files()

    def load_files(self):
        self.current_spec_filename = self.files[self.current_file_index]
        self.current_mask_filename = self.current_spec_filename.replace('__spec.npy', '__mask.npy')
        self.current_spec_file = np.load(os.path.join(self.feature_path, self.current_spec_filename))
        self.current_mask_file = np.load(os.path.join(self.feature_path, self.current_mask_filename))
        self.total_indexes = int(self.current_spec_file.shape[0] / self.num_samples) - 1
        self.current_index = 0

    def get_next(self):
        if self.current_index > self.total_indexes:
            self.current_file_index += 1

            if self.current_file_index >= len(self.files):
                self.current_file_index = 0

            if self.current_spec_file.shape[0] % self.num_samples > 10:
                inputs = self.current_spec_file[self.current_spec_file.shape[0] - self.num_samples:, :]
                targets = self.current_mask_file[self.current_mask_file.shape[0] - self.num_samples:, :]

                self.load_files()

                return inputs, targets

            self.load_files()

        start_index = self.current_index * self.num_samples
        end_index = (self.current_index + 1) * self.num_samples
        inputs = self.current_spec_file[start_index:end_index, :]
        targets = self.current_mask_file[start_index:end_index, :]
        self.current_index += 1

        return inputs, targets

    def get_dataset_len(self):
        total = 0

        for filename in self.files:
            arr = np.load(os.path.join(self.feature_path, filename))
            total = total + int(arr.shape[0] / self.num_samples)

            if arr.shape[0] % self.num_samples > 10:
                total += 1

        return total

    def get_validation(self):
        inputs = []
        targets = []

        for _ in range(self.batch_size):
            inp, tar = self.get_next()
            inputs.append(inp.flatten())
            targets.append(tar.flatten())

        return np.array(inputs), np.array(targets)

    def __len__(self):
        return int(self.dataset_len / self.batch_size)

    def __getitem__(self, idx):
        inputs = []
        targets = []

        for _ in range(self.batch_size):
            inp, tar = self.get_next()
            inputs.append(inp.flatten())
            targets.append(tar.flatten())

        return np.array(inputs), np.array(targets)


def readAudioScipy(filein):
    sampleRate, audioObj = wavfile.read(filein)
    bitrate = audioObj.dtype

    try:
        maxv = np.finfo(bitrate).max
    except:
        maxv = np.iinfo(bitrate).max
    return audioObj.astype('float')/maxv, sampleRate, bitrate


n_window = 1024
n_overlap = 256
ham_win = np.hamming(n_window)


class SingleAudio():
    def __init__(self, audio_path):
        self.mixed_spectrogram = self.generate_spectrogram(audio_path)
        self.num_samples = 20
        self.padded_spectrogram = np.pad(
            self.mixed_spectrogram,
            ((self.num_samples - 1, self.num_samples - 1), (0, 0)),
            'constant'
        )
        self.summed_mask = np.zeros(self.padded_spectrogram.shape)

    def generate_spectrogram(self, audio_path):
        audioObj, sampleRate, bitrate = readAudioScipy(audio_path)
        audio = audioObj[:,0] + audioObj[:,1]

        [f, t, mixed_spec] = spectral.spectrogram(
            x=audio,
            window=ham_win,
            nperseg=n_window,
            noverlap=n_overlap,
            detrend=False,
            return_onesided=True,
            mode='magnitude')

        return mixed_spec.T

    def __len__(self):
        return self.padded_spectrogram.shape[0] - self.num_samples

    def __getitem__(self, idx):
        return self.padded_spectrogram[idx:idx + self.num_samples, :].flatten()

    def update_matrix(self, flattened_mask, idx):
        mask = np.reshape(flattened_mask[0], (self.num_samples, 513))

        for i in range(self.num_samples):
            for j in range(513):
                self.summed_mask[idx + i][j] += mask[i][j]

    def generate_masks(self, threshold):
        mask_normalized = self.summed_mask[19:self.padded_spectrogram.shape[0] - 19] / self.num_samples
        voice_mask = np.zeros(mask_normalized.shape)
        noise_mask = np.zeros(mask_normalized.shape)

        for i in range(mask_normalized.shape[0]):
            for j in range(mask_normalized.shape[1]):
                if mask_normalized[i][j] > threshold:
                    voice_mask[i][j] = 1

                if mask_normalized[i][j] < (1 - threshold):
                    noise_mask[i][j] = 1

        return voice_mask, noise_mask

    def get_spectrograms(self, threshold):
        voice_mask, noise_mask = self.generate_masks(threshold)
        # print(voice_mask)
        # print(noise_mask)
        voice = self.mixed_spectrogram * voice_mask
        noise = self.mixed_spectrogram * noise_mask

        return voice, noise

    def plot_spectrograms(self, threshold=0.5):
        voice, noise = self.get_spectrograms(threshold)

        plt.figure(figsize=(18, 16), dpi= 60, facecolor='w', edgecolor='k')
        specshow(voice, sr=16000, x_axis='time', y_axis='hz', x_coords=np.linspace(0, 1, voice.shape[1]))
        plt.xlabel("Time (s)")
        plt.title("Clean Spectrogram", fontsize=14)
        # plt.colorbar(format='%+02.0f dB')
        # plt.savefig('clean_predicted.png')
        plt.show()
        # plt.clf()

        # plt.figure(figsize=(18, 16), dpi= 60, facecolor='w', edgecolor='k')
        # specshow(noise, sr=16000, x_axis='time', y_axis='hz', x_coords=np.linspace(0, 1, noise.shape[1]))
        # plt.xlabel("Time (s)")
        # plt.title("Noise Spectrogram", fontsize=14)
        # plt.colorbar(format='%+02.0f dB')
        # plt.savefig('noise_predicted.png')
