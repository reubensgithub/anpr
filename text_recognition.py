import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from builtins import ValueError
from PIL import Image
import numpy as np
import cv2
import os
from sklearn.metrics import average_precision_score


# limits GPU VRAM consumption growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#train_dataset_df = pd.read_csv('tr_dataset/train/_annotations.csv')
#validation_dataset_df = pd.read_csv('tr_dataset/valid/_annotations.csv')
#test_dataset_df = pd.read_csv('tr_dataset/test/_annotations.csv')
train_dataset_df = pd.read_csv('/content/drive/My Drive/tr_dataset/train/_annotations.csv')
validation_dataset_df = pd.read_csv('/content/drive/My Drive/tr_dataset/valid/_annotations.csv')
test_dataset_df = pd.read_csv('/content/drive/My Drive/tr_dataset/test/_annotations.csv')


'''train_dataset_df = train_dataset_df.sample(frac=1/50, random_state=42)
validation_dataset_df = validation_dataset_df.sample(frac=1/50, random_state=42)
test_dataset_df = test_dataset_df.sample(frac=1/50, random_state=42)'''

#train_dataset_path = 'tr_dataset/train/'
#validation_dataset_path = 'tr_dataset/valid/'
#test_dataset_path = 'tr_dataset/test/'

train_dataset_path = '/content/drive/My Drive/tr_dataset/train/'
validation_dataset_path = '/content/drive/My Drive/tr_dataset/valid/'
test_dataset_path = '/content/drive/My Drive/tr_dataset/test/'

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

def read_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    return image


def get_train_image_path(img_name):
    return train_dataset_path + img_name

def get_validation_image_path(img_name):
    return validation_dataset_path + img_name

def get_test_image_path(img_name):
    return test_dataset_path + img_name

def connect_train_dataset_annotations(row, target_size=(256, 256)):
    image_path = get_train_image_path(row['filename'])
    image = read_image(image_path)
    image = cv2.resize(image, target_size)
    xmin = int(row['xmin'] * 0.4)
    ymin = int(row['ymin'] * 0.4)
    xmax = int(row['xmax'] * 0.4)
    ymax = int(row['ymax'] * 0.4)
    return image, (xmin, ymin, xmax, ymax)

def connect_validation_dataset_annotations(row, target_size=(256, 256)):
    image_path = get_validation_image_path(row['filename'])
    image = read_image(image_path)
    image = cv2.resize(image, target_size)
    xmin = int(row['xmin'] * 0.4)
    ymin = int(row['ymin'] * 0.4)
    xmax = int(row['xmax'] * 0.4)
    ymax = int(row['ymax'] * 0.4)
    return image, (xmin, ymin, xmax, ymax)

def connect_test_dataset_annotations(row, target_size=(256, 256)):
    image_path = get_test_image_path(row['filename'])
    image = read_image(image_path)
    image = cv2.resize(image, target_size)
    xmin = int(row['xmin'] * 0.4)
    ymin = int(row['ymin'] * 0.4)
    xmax = int(row['xmax'] * 0.4)
    ymax = int(row['ymax'] * 0.4)
    return image, (xmin, ymin, xmax, ymax)


def custom_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), diff.dtype)
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def calculate_iou(y_true, y_pred, iou_threshold=0.5):
    true_xmin, true_ymin, true_xmax, true_ymax = tf.split(tf.cast(y_true, tf.float32), 4, axis=-1)
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = tf.split(tf.cast(y_pred, tf.float32), 4, axis=-1)

    # Calculate coordinates of intersection area
    xmin = tf.maximum(true_xmin, pred_xmin)
    ymin = tf.maximum(true_ymin, pred_ymin)
    xmax = tf.minimum(true_xmax, pred_xmax)
    ymax = tf.minimum(true_ymax, pred_ymax)

    intersection_width = tf.maximum(0.0, tf.minimum(true_xmax, pred_xmax) - tf.maximum(true_xmin, pred_xmin))
    intersection_height = tf.maximum(0.0, tf.minimum(true_ymax, pred_ymax) - tf.maximum(true_ymin, pred_ymin))
    intersection_area = intersection_width * intersection_height

    # Calculate areas of true and predicted bounding boxes
    true_area = (true_xmax - true_xmin) * (true_ymax - true_ymin)
    pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)

    # Calculate Union area
    union_area = true_area + pred_area - intersection_area

    # Avoid division by zero
    epsilon = 1e-7
    iou = intersection_area / (union_area + epsilon)

    iou = tf.where(iou >= iou_threshold, 1.0, 0.0)

    return iou



def calculate_mAP(y_true, y_pred):
    mAP = average_precision_score(y_true, y_pred)
    return mAP


train_dataset = []
for index, row in train_dataset_df.iterrows():
    train_dataset.append(connect_train_dataset_annotations(row))

validation_dataset = []
for index, row in validation_dataset_df.iterrows():
    validation_dataset.append(connect_validation_dataset_annotations(row))

test_dataset = []
for index, row in test_dataset_df.iterrows():
    test_dataset.append(connect_test_dataset_annotations(row))

train_images = np.array([item[0] for item in train_dataset])
train_labels = np.array([item[1] for item in train_dataset])

validation_images = np.array([item[0] for item in validation_dataset])
validation_labels = np.array([item[1] for item in validation_dataset])

test_images = np.array([item[0] for item in test_dataset])
test_labels = np.array([item[1] for item in test_dataset])


def create_model(input_shape=(256, 256, 3)):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3,3), 1, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D())

    model.add(tf.keras.layers.Conv2D(64, (3,3), 1, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(padding='same'))

    model.add(tf.keras.layers.Conv2D(128, (3,3), 1, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(padding='same'))

    model.add(tf.keras.layers.Conv2D(128, (3,3), 1, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(padding='same'))

    model.add(tf.keras.layers.Conv2D(128, (3,3), 1, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(padding='same'))

    model.add(tf.keras.layers.Conv2D(128, (3,3), 1, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(padding='same'))

    model.add(tf.keras.layers.Conv2D(256, (3,3), 1, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(padding='same'))

    model.add(tf.keras.layers.Conv2D(256, (3,3), 1, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(padding='same'))

    model.add(tf.keras.layers.Conv2D(256, (3,3), 1, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(padding='same'))

    model.add(tf.keras.layers.Conv2D(256, (3,3), 1, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(padding='same'))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(4, activation='linear')) # 4 corners of a recognised number plate region

    return model

model = create_model()

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',  # Monitor validation loss
    factor=0.1,           # Factor by which the learning rate will be reduced
    patience=3,           # Number of epochs with no improvement after which learning rate will be reduced
    min_lr=1e-7           # Minimum learning rate
)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=custom_loss, metrics=[calculate_iou, 'accuracy'])

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_dataset_df,
    directory=train_dataset_path,
    x_col="filename",
    y_col=["xmin", "ymin", "xmax", "ymax"],
    target_size=(256,256),
    batch_size=32,
    class_mode="raw"
)

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=validation_dataset_df,
    directory=validation_dataset_path,
    x_col="filename",
    y_col=["xmin", "ymin", "xmax", "ymax"],
    target_size=(256,256),
    batch_size=32,
    class_mode="raw"
)

hist = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[tensorboard_callback, lr_scheduler])

'''iou_scores = []
for i in range(len(validation_images)):
    predictions = model.predict(np.array([validation_images[i]]))
    iou = calculate_iou(validation_labels[i], predictions[0])
    iou_scores.append(iou)

average_iou = np.mean(iou_scores)
print("Average IoU:", average_iou)'''

#model.evaluate()
print("Model training done! ")
model.save('/content/drive/My Drive/tr_model1')

import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


def calculate_iou(y_true, y_pred, iou_threshold=0.5):
    true_xmin, true_ymin, true_xmax, true_ymax = tf.split(tf.cast(y_true, tf.float32), 4, axis=-1)
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = tf.split(tf.cast(y_pred, tf.float32), 4, axis=-1)

    # Calculate coordinates of intersection area
    xmin = tf.maximum(true_xmin, pred_xmin)
    ymin = tf.maximum(true_ymin, pred_ymin)
    xmax = tf.minimum(true_xmax, pred_xmax)
    ymax = tf.minimum(true_ymax, pred_ymax)

    intersection_width = tf.maximum(0.0, tf.minimum(true_xmax, pred_xmax) - tf.maximum(true_xmin, pred_xmin))
    intersection_height = tf.maximum(0.0, tf.minimum(true_ymax, pred_ymax) - tf.maximum(true_ymin, pred_ymin))
    intersection_area = intersection_width * intersection_height

    # Calculate areas of true and predicted bounding boxes
    true_area = (true_xmax - true_xmin) * (true_ymax - true_ymin)
    pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)

    # Calculate Union area
    union_area = true_area + pred_area - intersection_area

    # Avoid division by zero
    epsilon = 1e-7
    iou = intersection_area / (union_area + epsilon)

    iou = tf.where(iou >= iou_threshold, 1.0, 0.0)

    return iou


def custom_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), diff.dtype)
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss


model = tf.keras.models.load_model('/content/drive/My Drive/tr_model', custom_objects={'custom_loss': custom_loss,
                                                                                       'calculate_iou': calculate_iou})  # Adjust the path as needed


def visualize_predictions(images, labels, predictions):
    num_images = len(images)

    for i in range(num_images):
        image = images[i]
        true_box = labels[i]
        pred_box = predictions[i]

        # Convert bounding box coordinates to integers
        true_xmin, true_ymin, true_xmax, true_ymax = true_box.astype(int)
        pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_box.astype(int)

        # Draw true bounding box
        cv2.rectangle(image, (true_xmin, true_ymin), (true_xmax, true_ymax), (0, 255, 0), 2)

        # Draw predicted bounding box
        cv2.rectangle(image, (pred_xmin, pred_ymin), (pred_xmax, pred_ymax), (255, 0, 0), 2)

        # Show image with bounding boxes
        plt.imshow(image)
        plt.axis('off')
        plt.show()


# Use the trained model to make predictions
test_predictions = model.predict(test_images)

# Visualize predictions on a subset of test images
subset_size = 50  # Adjust the subset size as needed
visualize_predictions(test_images[:subset_size], test_labels[:subset_size], test_predictions[:subset_size])


