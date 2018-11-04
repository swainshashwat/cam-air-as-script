import cv2
import numpy as np
import imutils

def main_preprocess(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28,28),
     interpolation=cv2.INTER_AREA)
    image = cv2.bitwise_not(image)
    image = cv2.equalizeHist(image)
    
    return image.reshape(-1,28,28,1)