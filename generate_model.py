import tensorflow as tf
#from tensorflow.keras.models import Conv2D
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
#from matplotlib import pyplot as plt

final_dataset = 'C:/Users/reube/Desktop/final_dataset'
test_dataset = 'C:/Users/reube/Desktop/test_dataset'
#preprocessed = 'C:/Users/reube/Downloads/anpr-dataset/preprocessed'

# limits GPU VRAM consumption growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

dataset = tf.keras.utils.image_dataset_from_directory(test_dataset)
#dataset_iterator = dataset.as_numpy_iterator()
#batch = dataset_iterator.next()
#print(batch[1])
dataset = dataset.map(lambda x, y: (x/255, y))
scaled_iterator = dataset.as_numpy_iterator()
batch = scaled_iterator.next()
print("Minimum of batch is: ", batch[0].min())
print("Maximum of batch is: ", batch[0].max())
print("Length of dataset: ", len(dataset))
# 755 batches of size 32

train_size = int(len(dataset) * 0.7)
validation_size = int(len(dataset) * 0.2)
test_size = int(len(dataset) * 0.1)+1

print("Train size: ", train_size) # 528
print("Validation size: ", validation_size) # 151
print("Test size: ", test_size) # 76

train = dataset.take(train_size)
val = dataset.skip(train_size).take(validation_size)
test = dataset.skip(train_size + validation_size).take(test_size)

model = tf.keras.models.Sequential()

#adding a convolutional layer and max pooling layer
model.add(tf.keras.layers.Conv2D(16, (3,3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(32, (3,3), 1, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(16, (3,3), 1, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

'''fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()'''

model.save('models')


