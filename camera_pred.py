import cv2 as cv # for image processing and input
import numpy as np # For matrix calculations
import math
import time
from skimage.feature import hog
from sklearn.externals import joblib
from collections import deque # for center points to be used later on

# capturing video using the default camera
cap = cv.VideoCapture(0)

# centerpoints are used in order to plot the data on the new board from image
centerpoints = deque()

# assigning result some temporary negative values
result = [-1, -2]

# Defining the intensity of green color to be identified.
# It is in HSV format
sensitivity = 20
lgreen = np.array([60-sensitivity, 100, 50])
ugreen = np.array([60+sensitivity, 255, 255])

# Black-Board where identified image from video is to be drawn
board = np.zeros((300, 300), dtype='uint8')

# importing classifier and preprocessed data from the earlier saved model
clf, preproc = joblib.load('mnist_clf_svm.pkl')

# Function to return the contours within the image
def get_ict(roi):  
    roi = cv.GaussianBlur(roi, (7, 7), 0) # it softens the image thus making it a bit blurry
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV) # converting to HSV color format

    #filtering out the color 
    roi_range = cv.inRange(hsv_roi, lgreen, ugreen)

    # Finding contours based on the detected green color
    _, contour, hier = cv.findContours(
        roi_range.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    return roi, contour, roi_range

# main() starts:
while cap.isOpened():
	# read each and every frame
    _, frame = cap.read()

    frame = cv.flip(frame, 1)

    #making rectangle to mark the region of interest which will be detected by board
    cv.rectangle(frame, (500, 50), (800, 350), (255, 0, 0), 2)

    roi = frame[50:350, 500:800, :] # ROI : Region of Interest
    roi, contours, roi_range = get_ict(roi)

    draw_started = False
    draw_stopped = False

    # confirming if any contour exists or not
    if len(contours)>0:
        draw_started = True

        # Finding the largest contour based on area
        max_contour = max(contours, key=cv.contourArea)

        # marks the center of moment of the contour detected
        M = cv.moments(max_contour)

        # using the center of moment to find put the centroid of the contour
        try:
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        except:
            continue

        # center obtained is appended to the deque
        centerpoints.appendleft(center)
    else:
        draw_stopped= True

    # marking the detected points on the ROI image and then plotting on the board
    for i in range(1, len(centerpoints)):

    	# distance between the two nearest detected points
        if math.sqrt((centerpoints[i-1][0] - centerpoints[i][0])**2 + (centerpoints[i-1][1] - centerpoints[i][1])**2) < 50:

            cv.line(roi, centerpoints[i-1], centerpoints[i], (200, 200, 200), 5, cv.LINE_AA)
            cv.line(board, (centerpoints[i-1][0]+15, centerpoints[i-1][1]+15), (centerpoints[i][0]+15, centerpoints[i][1]+15),
                    255, 7, cv.LINE_AA)

    # resizing the board to fit image in the classifier
    img = cv.resize(board, (28, 28))

    if np.max(board) != 0 and draw_started== True and draw_stopped== True:
    	# works like erosion and dilusion. Enchances white and dark according to values passed.
        img = cv.morphologyEx(img, cv.MORPH_OPEN, (5, 5))
        board = cv.morphologyEx(board, cv.MORPH_OPEN, (5, 5))

        draw_started = False
        draw_stopped = False

    # If something green is detected on the board:
    if np.max(board) != 0:
        img = np.array(img)

        hog_features = hog(img, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), block_norm='L2')
        hog_features = preproc.transform(np.array([hog_features], 'float64'))

        # predicting the final result
        result = clf.predict(hog_features)
    
    if int(result[0])<0:
        cv.putText(frame, "Predicted number: " , (5, 420),
                   cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)
    else:
        cv.putText(frame, "Predicted number: " + str(int(result)), (5, 420),
                   cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)
    
    # displays the images as windows   
    cv.imshow('frame', frame)
    #cv.imshow('roi', roi)
    #cv.imshow('roi_image', roi_range)
    cv.imshow('board', board)
    cv.imshow('img', img)

    # waitkey waits for the input before processing next frame
    k = cv.waitKey(1) & 0xFF
    if k == ord('q'):
    	# Quit when 'Q' is pressed
        break

    #clear the board and result when 'c' is pressed
    elif k == ord('c'):
        board.fill(0)
        centerpoints.clear()
        result = [-2, -1]
    
# Destroying the windows after end of program    
cap.release()
cv.destroyAllWindows()


