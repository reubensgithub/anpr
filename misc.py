import os
import cv2

# Function to resize images
def resize_images(dataset_path):
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg") or filename.endswith(".png"): # Assuming images are in jpg or png format
            image_path = os.path.join(dataset_path, filename)
            img = cv2.imread(image_path)
            img = cv2.resize(img, (256, 256))
            cv2.imwrite(image_path, img)

# Resize images in train dataset
resize_images(train_dataset_path)

# Resize images in validation dataset
resize_images(validation_dataset_path)

# Resize images in test dataset
resize_images(test_dataset_path)
