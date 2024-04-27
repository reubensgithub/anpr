import cv2
import imutils
import numpy as np
import time
import os
import shutil


def preprocess_image(image_path):
    #grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_path = cv2.imread(image_path)
    image_path = cv2.resize(image_path, (640, 640))
    if image_path is None:
        print("Error: Image not found")
        return None
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
    #mask_combined = cv2.bitwise_or(mask_white, mask_yellow)
    color_regions = cv2.bitwise_and(image_path, image_path, mask=mask_combined)
    grey = cv2.cvtColor(color_regions, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(grey, 11, 17, 17)
    edged = cv2.Canny(filtered, 10, 200)
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 6)
    #thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 55, 20)
    cv2.imshow("thresh", thresh)
    cv2.waitKey(0)
    kernel = np.ones((5,5), np.uint8)
    points = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_points = imutils.grab_contours(points)
    contour_points = sorted(contour_points, key=cv2.contourArea, reverse=True)
    min_aspect_ratio = 2.0
    max_aspect_ratio = 5.0

    #cv2.imshow('Enhanced Image', filtered)
    #cv2.imshow('Edged Image', edged)
    #cv2.imshow('Thresh Image', thresh)
    #cv2.waitKey(0)

    contour_image = image.copy()
    cv2.drawContours(contour_image, contour_points, -1, (0, 255, 0), 2)
    cv2.imshow("All Contours", contour_image)
    cv2.waitKey(0)

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
                    if has_text_inside(contour, thresh):
                        new_image = image[y:y + h, x:x + w]
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
                        if has_text_inside(contour, thresh):
                            new_image = image[y:y+h, x:x+w]
                            return new_image
                else:
                    points = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contour_points = imutils.grab_contours(points)
                    contour_points = sorted(contour_points, key=cv2.contourArea, reverse=True)

def has_text_inside(contour, thresh):
    x, y, w, h = cv2.boundingRect(contour)
    roi = thresh[y:y + h, x:x + w]
    total_pixels = np.prod(roi.shape[:2])
    white_pixels = total_pixels - cv2.countNonZero(roi)
    return white_pixels > total_pixels * 0.4


'''def preprocess_all(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = os.listdir(input_folder)

    for image_name in images:
        if image_name.endswith((".jpg", '.jpeg', '.png')):
            image_path = os.path.join(input_folder, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                preprocessed_image = preprocess_image(image)
                if preprocessed_image is not None:
                    output_path = os.path.join(output_folder, image_name)
                    cv2.imwrite(output_path, preprocessed_image)

                    print(f"Preprocessed image saved: {output_path}")
                else:
                    print(f"Failed to preprocess image: {image}")
            else:
                print(f"Unable to read image: {image}")


def put_img_into_folder(folder_path):
    files = os.listdir(folder_path)

    for file in files:
        if file.lower().endswith(('.png','.jpg','.jpeg')):
            file_name = os.path.splitext(file)[0]

            subfolder_path = os.path.join(folder_path, file_name)
            if not os.path.exists(subfolder_path):
                os.mkdir(subfolder_path)

            shutil.move(os.path.join(folder_path, file), subfolder_path)


def rename_subfolders(folder_path):
    subfolders = [subfolder for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]

    for subfolder in subfolders:
        old_path = os.path.join(folder_path, subfolder)
        new_name = subfolder[:4] + ' ' + subfolder[4:]
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)

def transfer_subfolders(source_folder, destination_folder):
    subfolders = [subfolder for subfolder in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, subfolder))]

    for subfolder in subfolders:
        source_path = os.path.join(source_folder, subfolder)
        destination_path = os.path.join(destination_folder, subfolder)

        shutil.copytree(source_path, destination_path)

yellow_synthetic_uk_plates = 'C:/Users/reube/Downloads/yellowplate_augmented'
white_synthetic_uk_plates = 'C:/Users/reube/Downloads/whiteplate_augmented'

input_folder = 'C:/Users/reube/Downloads/anpr-dataset/images'
input_folder2 = 'C:/Users/reube/Downloads/archive/images'
output_folder = 'C:/Users/reube/Downloads/anpr-dataset/preprocessed'
output_folder2 = 'C:/Users/reube/Downloads/archive/preprocessed'

final_destination = 'C:/Users/reube/Desktop/final_dataset'

## all preprocessed images are currently in 'C:/Users/reube/Downloads/anpr-dataset/preprocessed'

#print("PROGRAM IS DONE EXECUTING")'''


image = cv2.imread('C:/Users/reube/Downloads/DSC06005.jpg')
image = cv2.resize(image, (640, 640))
new_image = preprocess_image('C:/Users/reube/Downloads/DSC06005.jpg')
#new_image = cv2.resize(new_image, (720, 480))
cv2.imshow('Original image', image)
cv2.waitKey(0)
cv2.imshow('New image', new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# NEXT STEPS INVOLVE ACTUALLY TRAINING THE MODELS. USE CNNS (Convolutional Neural Networks).