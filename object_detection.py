import tensorflow as tf
import pandas as pd
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

train_dataset_df = pd.read_csv('od_dataset1/train/_annotations.csv')
validation_dataset_df = pd.read_csv('od_dataset1/valid/_annotations.csv')
test_dataset_df = pd.read_csv('od_dataset1/test/_annotations.csv')

train_dataset_path = 'od_dataset1/train/'
validation_dataset_path = 'od_dataset1/valid/'
test_dataset_path = 'od_dataset1/test/'

'''def resize_images(dataset_path):
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(dataset_path, filename)
            img = cv2.imread(image_path)
            img = cv2.resize(img, (640, 640))
            cv2.imwrite(image_path, img)

resize_images(train_dataset_path)
resize_images(validation_dataset_path)
resize_images(test_dataset_path)'''

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
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

def read_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256,256))
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


def calculate_iou(y_true, y_pred):
    true_xmin, true_ymin, true_xmax, true_ymax = tf.split(tf.cast(y_true, tf.float32), 4, axis=-1)
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = tf.split(tf.cast(y_pred, tf.float32), 4, axis=-1)

    # Calculate coordinates of intersection area
    xmin = tf.maximum(true_xmin, pred_xmin)
    ymin = tf.maximum(true_ymin, pred_ymin)
    xmax = tf.minimum(true_xmax, pred_xmax)
    ymax = tf.minimum(true_ymax, pred_ymax)

    intersection_area = tf.maximum(xmax - xmin, 0) * tf.maximum(ymax - ymin, 0)

    # Calculate areas of true and predicted bounding boxes
    true_area = (true_xmax - true_xmin) * (true_ymax - true_ymin)
    pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)

    # Calculate Union area
    union_area = true_area + pred_area - intersection_area

    # Avoid division by zero
    epsilon = 1e-7
    iou = intersection_area / (union_area + epsilon)

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

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='linear')  # Output 4 values for bounding box coordinates
])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=custom_loss, metrics=[calculate_iou, 'accuracy'])

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_dataset_df,
    directory=train_dataset_path,
    x_col="filename",
    y_col=["xmin", "ymin", "xmax", "ymax"],
    target_size=(256,256),
    batch_size=16,
    class_mode="raw"
)

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=validation_dataset_df,
    directory=validation_dataset_path,
    x_col="filename",
    y_col=["xmin", "ymin", "xmax", "ymax"],
    target_size=(256,256),
    batch_size=16,
    class_mode="raw"
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_dataset_df,
    directory=test_dataset_path,
    x_col="filename",
    y_col=["xmin", "ymin", "xmax", "ymax"],
    target_size=(256,256),
    batch_size=16,
    class_mode="raw"
)

hist = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[tensorboard_callback])


iou_scores = []
for i in range(len(validation_images)):
    predictions = model.predict(np.array([validation_images[i]]))
    iou = calculate_iou(validation_labels[i], predictions[0])
    iou_scores.append(iou)

average_iou = np.mean(iou_scores)
print("Average IoU:", average_iou)

model.evaluate()
model.save('od_model')


