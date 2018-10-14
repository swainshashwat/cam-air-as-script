
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from collections import deque


# In[2]:


# Define the upper and lower boundaries for a color to be considered "Blue"
blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])

# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)


# In[10]:


# initialize deques
bPoints = [deque(maxlen=512)]
gPoints = [deque(maxlen=512)]
rPoints = [deque(maxlen=512)]
yPoints = [deque(maxlen=512)]

# indexes
bIndex = 0
gIndex = 0
rIndex = 0
yIndex = 0


# In[11]:


# An array and an index variable to get the color-of-interest
# B G R Y
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0


# In[14]:


# create a blank white image
paintWindow = np.zeros((471, 636, 3)) + 255


# In[17]:


# draw rectangular buttons on the white image
paintWindow = cv2.rectangle(paintWindow,
                            pt1=(40, 1), pt2=(140, 65), color=(0,0,0),
                            thickness=2)
paintWindow = cv2.rectangle(paintWindow,
                            pt1=(160, 1), pt2=(255, 65), color=colors[0],
                            thickness=-1)
paintWindow = cv2.rectangle(paintWindow,
                            pt1=(275, 1), pt2=(370, 65), color=colors[1],
                            thickness=-1)
paintWindow = cv2.rectangle(paintWindow,
                            pt1=(390, 1), pt2=(485, 65), color=colors[2],
                            thickness=-1)
paintWindow = cv2.rectangle(paintWindow,
                            pt1=(505, 1), pt2=(600, 65), color=colors[3],
                            thickness=-1)


# In[ ]:


# Label the rectangular boxes drawn on the image
cv2.putText(paintWindow, "CLEAR", org=(49, 33),
            fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.5,
           color=(0,0,0), thickness=2, lineType=cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", org=(185, 33),
            fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.5,
           color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", org=(298, 33),
            fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.5,
           color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
cv2.putText(paintWindow, "RED", org=(420, 33),
            fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.5,
           color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", org=(520, 33),
            fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.5,
           color=(150,150,150), thickness=2, lineType=cv2.LINE_AA)

# initializing camera
camera = cv2.VideoCapture(0)

while True:
    # Grab the current paintWindow
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Hue Saturation Value

    if not grabbed: # when we reach the eof of video files
        break
    
    # Add paint interface to the camera feed captured through the webcam
    frame = cv2.rectangle(frame, pt1=(40,1), pt2=(140,65),
     color=(122,122,122), thickness=2)
    frame = cv2.rectangle(frame, pt1=(160,1), pt2=(255,65),
     color=colors[0], thickness=2)
    frame = cv2.rectangle(frame, pt1=(275,1), pt2=(370,65),
     color=colors[1], thickness=2)
    frame = cv2.rectangle(frame, pt1=(390,1), pt2=(485,65),
     color=colors[2], thickness=2)
    frame = cv2.rectangle(frame, pt1=(505,1), pt2=(600,65),
     color=colors[3], thickness=2)
    
    cv2.putText(frame, "CLEAR", org=(49, 33),
            fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.5,
           color=(0,0,0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(frame, "BLUE", org=(185, 33),
                fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.5,
            color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(frame, "CLEAR", org=(298, 33),
                fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.5,
            color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(frame, "CLEAR", org=(420, 33),
                fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.5,
            color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", org=(520, 33),
                fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.5,
            color=(150,150,150), thickness=2, lineType=cv2.LINE_AA)

    # determine which pixels fall within the blue boundaries &
    # then blur binary image
    blueMask = cv2.inRange(hsv, blueLower, blueUpper)
    blueMask = cv2.erode(blueMask, kernel, iterations=1)

    # finding contours in the image
    (_, cnts, _) = cv2.findContours(blueMask.copy(),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # check to see if any contours (blue stylus) was found
    if len(cnts) > 0:

        # find the largest contour
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

        # radius of the enclosing circle around the found contour
        ((x,y), radius) = cv2.minEnclosingCircle(cnt)

        # Draw the circle around the contour
        cv2.circle(frame, (int(x), int(y)), int(radius),
         (0, 255,255), thickness=2)

        # Get the moments to calculate the center of the contour( circle )
        M = cv2.moments(cnt)

        if M['m00']!=0:
            center = (int(M['m10']/ M['m00']), int(M['m10']/ M['m00']))
        else:
            center = (int(M['m10']/ 1), int(M['m10']/ 1))
        # drawing using the blue center
        if center[1] <= 65:
            
            # Clear All
            if 40 <= center[0] < 140:
                
                # Emplty all the point holders
                bPoints = [deque(maxlen=512)]
                gPoints = [deque(maxlen=512)]
                rPoints = [deque(maxlen=512)]
                yPoints = [deque(maxlen=512)]

                # resetting the indexes
                bIndex = 0
                gIndex = 0
                rIndex = 0
                yIndex = 0

                # Making the frame White again
                paintWindow[67:,:,:] = 255
        # when the contour center touches a box,
        #  assign its appropriate color

        elif 160 <= center[0] <=255:  # Blue Box
            colorIndex = 0
        elif 275 <= center[0] <=370:  # Green Box
            colorIndex = 1
        elif 390 <= center[0] <=485:  # Red Box
            colorIndex = 2
        elif 505 <= center[0] <=600:  # Yellow Box
            colorIndex = 3

        # Store the center (point) in its assigned color deque
        else:
            if colorIndex==0:
                bPoints[bIndex].appendleft(center)
            elif colorIndex==1:
                gPoints[gIndex].appendleft(center)
            elif colorIndex==2:
                rPoints[rIndex].appendleft(center)
            elif colorIndex==3:
                yPoints[yIndex].appendleft(center)

    # Draw lines of all the colors (BGRY)
    points = [bPoints, gPoints, rPoints, yPoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k-1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k-1], points[i][j][k],
                    colors[i], thickness=2)
                cv2.line(paintWindow, points[i][j][k-1], points[i][j][k],
                    colors[i], thickness=2)
    
    # show the frame and the painWindow image
    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)
    cv2.imshow('HSV', hsv)

    # 'q' = QUIT
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()