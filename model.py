from sklearn.svm import LinearSVC
import numpy as np
import cv2 as cv
from PIL import Image  # Ensure Pillow is installed

class Model:

    def __init__(self):
        self.model = LinearSVC()

    def train_model(self, counters):
        img_list = []
        class_list = []

        # Read images from class 1
        for i in range(1, counters[0]):
            img = cv.imread(f'1/frame{i}.jpg', cv.IMREAD_GRAYSCALE)  # Read as grayscale
            img = cv.resize(img, (120, 140))  # Resize to 120x140, or any size that fits 16800 pixels
            img = img.reshape(16800)  # Flatten image
            img_list.append(img)
            class_list.append(1)

        # Read images from class 2
        for i in range(1, counters[1]):
            img = cv.imread(f'2/frame{i}.jpg', cv.IMREAD_GRAYSCALE)  # Read as grayscale
            img = cv.resize(img, (120, 140))  # Resize to 120x140
            img = img.reshape(16800)  # Flatten image
            img_list.append(img)
            class_list.append(2)

        # Convert lists to numpy arrays
        img_list = np.array(img_list)
        class_list = np.array(class_list)

        # Train the model
        self.model.fit(img_list, class_list)
        print("Model successfully trained!")

    def predict(self, frame):
        frame = frame[1]
        cv.imwrite("frame.jpg", cv.cvtColor(frame, cv.COLOR_RGB2GRAY))

        # Resize and save image with consistent dimensions
        img = Image.open("frame.jpg")
        img = img.resize((120, 140))  # Resize to 120x140
        img.save("frame.jpg")

        # Read, flatten, and predict
        img = cv.imread('frame.jpg', cv.IMREAD_GRAYSCALE)
        img = img.reshape(16800)  # Ensure the shape matches the training data
        prediction = self.model.predict([img])

        return prediction[0]
