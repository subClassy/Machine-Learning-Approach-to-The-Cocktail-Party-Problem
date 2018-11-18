import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam, SGD, Adadelta
from dataset import Dataset
import os

feature_path = '/home/ankur/Downloads/Others/features'
validation_path = '/home/ankur/Downloads/Others/validation'
validation_generator = Dataset(validation_path)
data_generator = Dataset(feature_path)
batch_size = data_generator.batch_size
num_samples = data_generator.num_samples
freq_bins = 513
units = num_samples * freq_bins

model = Sequential()
model.add(Dense(int(units / 2), activation='relu', input_dim=units))
model.add(Dense(int(units / 2), activation='sigmoid'))
model.add(Dense(units, activation='sigmoid', use_bias=False))

model.summary()

adam_o = adam(lr=0.0001)

epochs = 50

learning_rate = 0.01
decay_rate = learning_rate / epochs
sgd_o = SGD(lr=learning_rate, momentum=0.9, decay=decay_rate)
opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=1e-6)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit_generator(
    data_generator,
    epochs=epochs,
    steps_per_epoch=85,
    workers=3,
    use_multiprocessing=True,
    validation_data=validation_generator,
    validation_steps=10
)

X, Y = data_generator.get_validation()

scores = model.evaluate(x=X, y=Y, batch_size=batch_size)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

output_dir = '/home/ankur/Downloads/Others/models'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

model_json = model.to_json()

with open(os.path.join(output_dir, 'model5.json'), 'w') as json_file:
    json_file.write(model_json)

model.save_weights(os.path.join(output_dir, 'model5.h5'))
print('Saved model to disk')
