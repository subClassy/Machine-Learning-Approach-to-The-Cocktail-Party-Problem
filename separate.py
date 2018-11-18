import os
import numpy as np
from keras.models import model_from_json
from dataset import SingleAudio

output_dir = '/home/ankur/Downloads/Others/models'
json_file = open(os.path.join(output_dir, 'model4.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(os.path.join(output_dir, 'model4.h5'))
print('Loaded model from disk')

audio_path = '/home/ankur/Downloads/Others/mixed'
audio_name = '174-50561-0010__door_wood_knock_1-52290-A-30.wav'
single_audio = SingleAudio(os.path.join(audio_path, audio_name))

for i in range(len(single_audio)):
    x = np.array([single_audio[i]])
    print(x)
    predicted_mask = loaded_model.predict(x)
    print(predicted_mask)
    single_audio.update_matrix(predicted_mask, i)

print("Generating Spectrograms...")
single_audio.plot_spectrograms(0.66)
