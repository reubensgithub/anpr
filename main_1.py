import cv2
import imutils
import numpy as np
import time
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model, Sequential

def preprocess_image(image_path):
    if image_path is None:
        print("Error: Image not found")
        return None
    image_path = cv2.imread(image_path)
    image_path = cv2.resize(image_path, (640, 640))
    hsv = cv2.cvtColor(image_path, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([179, 30, 255])
    lower_gray = np.array([0, 0, 75])
    upper_gray = np.array([179, 50, 175])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
    mask_combined = cv2.bitwise_or(mask_white, cv2.bitwise_or(mask_gray, mask_yellow))
    color_regions = cv2.bitwise_and(image_path, image_path, mask=mask_combined)
    grey = cv2.cvtColor(color_regions, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(grey, 11, 17, 17)
    edged = cv2.Canny(filtered, 10, 200)
    #thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 6)
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 55, 20)
    points = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_points = imutils.grab_contours(points)
    contour_points = sorted(contour_points, key=cv2.contourArea, reverse=True)
    min_aspect_ratio = 2.0
    max_aspect_ratio = 5.0

    max_time = 5
    location = None
    start_time = time.time()
    while location is None and (time.time() - start_time) < max_time:
        for contour in contour_points:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                #location = approx
                x, y, w, h = cv2.boundingRect(approx)
                if w == 0 or h == 0:
                    continue
                aspect_ratio = float(w) / h
                #print("ASPECT RATIO IS: ", aspect_ratio)
                if min_aspect_ratio > aspect_ratio or aspect_ratio > max_aspect_ratio:
                    continue
                else:
                    new_image = image_path[y:y + h, x:x + w]
                    return new_image
            else:
                points = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contour_points = imutils.grab_contours(points)
                contour_points = sorted(contour_points, key=cv2.contourArea, reverse=True)
            if len(approx) != 4:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    #location = approx
                    x, y, w, h = cv2.boundingRect(approx)
                    if w == 0 or h == 0:
                        continue
                    aspect_ratio = float(w) / h
                    #print("ASPECT RATIO IS: ", aspect_ratio)
                    if min_aspect_ratio > aspect_ratio or aspect_ratio > max_aspect_ratio:
                        continue
                    else:
                        new_image = image_path[y:y+h, x:x+w]
                        return new_image
                else:
                    points = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contour_points = imutils.grab_contours(points)
                    contour_points = sorted(contour_points, key=cv2.contourArea, reverse=True)



#new_image = preprocess_image('All-newKiaSportage2.jpg')
#new_image = preprocess_image('DSC060051.jpg')
#new_image = preprocess_image('test.jpg') # output is numpy nd array with 3 dimensions (width, height, channels)
#cv2.imshow("New image", new_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

def create_tr_model():
    tr_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(36, activation='softmax')
    ])

    tr_model.load_weights('best_tr_model/best_model.h5')

def pipeline(image):
    number_plate = ''
    cropped = preprocess_image(image)
    cs_predictions = cs_model.predict(cropped)
    for box in cs_predictions:
        tr_predictions = tr_model.predict(box)
