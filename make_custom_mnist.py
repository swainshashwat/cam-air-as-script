import math
import sys
import os
import cv2
import scipy.misc
import numpy as np
from collections import deque

from my_preprocess import main_preprocess

# initializing result deque
center_points = deque()

# Define lower & upper range of hsv color to detect.
lower_blue = np.array([100, 80, 80])
upper_blue = np.array([160, 255, 255])

# blank window
paintWindow = np.zeros((471, 636, 3)) + 255

# initializing the camera
camera = cv2.VideoCapture(0)

while True:
    # Read and flip frame
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)
    
    paintWindow = np.zeros((471, 636, 3)) + 255

    # Blue the frame a little
    blur_frame = cv2.GaussianBlur(frame, (7, 7), 0)

    # Convert from BGR to HSV color format
    hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)

    # Blue here
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Make elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))

    # Opening morph (erosion followed by dilation)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find all contours
    contours, hierarchy = cv2.findContours(mask.copy(),
        cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    if len(contours) > 0:

        # Find the biggest contour
        biggest_contour = max(contours, key=cv2.contourArea)

        # Find center of contour and draw filled circle
        moments = cv2.moments(biggest_contour)
        center_of_contour = (int(moments['m10']/moments['m00']),
                            int(moments['m01']/moments['m00']))
        cv2.circle(frame, center_of_contour,
         radius=5, color=(0,0,255), thickness=-1)
        
        cv2.circle(paintWindow, center_of_contour,
         radius=5, color=(0,0,255), thickness=-1)

        # Bound the contour with circle
        ellipse = cv2.fitEllipse(biggest_contour)
        
        cv2.ellipse(frame, ellipse,
         color=(0, 255, 255), thickness=2)
        
        # save the centers of contour so we can draw through the frames
        # Stop drawing when the user presses 's'
        if cv2.waitKey(1) != ord("s"):
            center_points.appendleft(center_of_contour)
        else:
            None

    # Draw using the movement of contour centers
    for i in range(1, len(center_points)):
        pointer_color = (0, 0, 0)
        center_of_two_points = math.sqrt(
            ((center_points[i - 1][0] - center_points[i][0]) ** 2) +
            ((center_points[i - 1][1] - center_points[i][1]) ** 2))
        
        #print(center_of_two_points)
        
        if center_of_two_points <= 50:
            paintWindow = cv2.line(paintWindow, center_points[i-1], center_points[i],
                pointer_color, thickness=20)

    # clear the paintWindow
    if cv2.waitKey(1) == ord("c"):
        print('clearing the window...')
        center_points = deque()

    processed = main_preprocess(
        scipy.misc.toimage(
            paintWindow
        )
    )
    processed = scipy.misc.toimage(processed.reshape((28,28)))
    cv2.imshow('original', frame)
    cv2.imshow('paintWindow', paintWindow)
    cv2.imshow('Image', np.array(processed).reshape(28,28))



    # press 'q' to Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
    
        image_name = sys.argv[1]
        print('saving... '+image_name)
        cv2.imwrite(os.path.join('custom data' ,image_name+'.jpg'),
            paintWindow)

        print('closing...')
        
        break

    

# release resources
cv2.destroyAllWindows()
camera.release()