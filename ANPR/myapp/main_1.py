import cv2
import imutils
import numpy as np
import time
import functools
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model, Sequential

def license_plate_detection(image_path):
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

def compare(rect1, rect2):
        if abs(rect1[1] - rect2[1]) > 10:
            return rect1[1] - rect2[1]
        else:
            return rect1[0] - rect2[0]

def character_segmentation(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)
    _, labels = cv2.connectedComponents(thresh)
    mask = np.zeros(thresh.shape, dtype="uint8")
    image_pixels = image.shape[0] * image.shape[1]
    lb = image_pixels // 60
    ub = image_pixels // 20
    for (i, label) in enumerate(np.unique(labels)):
        if label == 0:
            continue
    
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        
        if numPixels > lb and numPixels < ub:
            mask = cv2.add(mask, labelMask)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare))
    return boundingBoxes

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
    cwd = os.getcwd()
    cwd = cwd.replace("\\", "/")
    cwd += '/myapp/'
    tr_model.load_weights(cwd + 'best_tr_model/best_model.h5')
    return tr_model


def pipeline(image):
    tr_model = create_tr_model()
    alphabet = ['0','1','2','3','4','5','6','7','8',
                '9','A','B','C','D','E','F','G','H',
                'I','J','K','L','M','N','O','P','Q',
                'R','S','T','U','V','W','X','Y','Z']
    number_plate = ''
    le = LabelEncoder()
    le.fit(alphabet)
    cropped = license_plate_detection(image)
    if cropped.shape[0] < 32 or cropped.shape[1] < 32:
        print("Valid number plate not found, breaking out")
        return None
    bounding_boxes = character_segmentation(cropped)
    
    cropped_characters = []
    for box in bounding_boxes:
        x,y,w,h = box
        crop = cropped[y:y+h, x:x+w]
        crop = cv2.resize(crop, (64, 64))
        cropped_characters.append(crop)

    cropped_characters = np.array(cropped_characters)
    cropped_characters = np.expand_dims(cropped_characters, axis=0)
    print(cropped_characters[0][0].shape)

    for char in cropped_characters[0]:
        char = np.expand_dims(char, axis=0)
        prediction = tr_model.predict(char)
        predicted_label = le.inverse_transform(np.argmax(prediction, axis=1))
        number_plate += predicted_label[0]

    return number_plate

#number_plate = pipeline('test.jpg')
#print(number_plate)

''' to get details, make a POST request:
'https://driver-vehicle-licensing.api.gov.uk/vehicle-enquiry/v1/vehicles'
API KEY = C3QKCXeyFO4N69PVcK1VB2anGN3wWjm74j02h0dC
The key is x-api-key
'''