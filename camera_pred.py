import cv2 as cv
import numpy as np 
import math
import time
from skimage.feature import hog
from sklearn.externals import joblib
from collections import deque

cap = cv.VideoCapture(0)
centerpoints = deque()
result = [-1, -2]

sensitivity = 20
lgreen = np.array([60-sensitivity, 100, 50])
ugreen = np.array([60+sensitivity, 255, 255])

board = np.zeros((300, 300), dtype='uint8')

clf, preproc = joblib.load('mnist_clf_svm.pkl')


def get_ict(roi):  # returns image, contours and threshold
    roi = cv.GaussianBlur(roi, (7, 7), 0)
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    roi_range = cv.inRange(hsv_roi, lgreen, ugreen)
    _, contour, hier = cv.findContours(
        roi_range.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    return roi, contour, roi_range

while cap.isOpened():
    _, frame = cap.read()

    frame = cv.flip(frame, 1)

    cv.rectangle(frame, (500, 50), (800, 350), (255, 0, 0), 2)
    roi = frame[50:350, 500:800, :] # ROI : Region of Interest
    roi, contours, roi_range = get_ict(roi)

    draw_started = False
    draw_stopped = False

    if len(contours)>0:
        draw_started = True
        max_contour = max(contours, key=cv.contourArea)

        M = cv.moments(max_contour)

        try:
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        except:
            continue
        # center obtained is appended to the deque
        centerpoints.appendleft(center)
    else:
        draw_stopped= True

    for i in range(1, len(centerpoints)):
        if math.sqrt((centerpoints[i-1][0] - centerpoints[i][0])**2 + (centerpoints[i-1][1] - centerpoints[i][1])**2) < 50:
            cv.line(roi, centerpoints[i-1], centerpoints[i], (200, 200, 200), 5, cv.LINE_AA)
            cv.line(board, (centerpoints[i-1][0]+15, centerpoints[i-1][1]+15), (centerpoints[i][0]+15, centerpoints[i][1]+15),
                    255, 7, cv.LINE_AA)

    img = cv.resize(board, (28, 28))

    if np.max(board) != 0 and draw_started== True and draw_stopped== True:
        img = cv.morphologyEx(img, cv.MORPH_OPEN, (5, 5))
        board = cv.morphologyEx(board, cv.MORPH_OPEN, (5, 5))

        draw_started = False
        draw_stopped = False

    if np.max(board) != 0:
        img = np.array(img)

        hog_features = hog(img, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), block_norm='L2')
        hog_features = preproc.transform(np.array([hog_features], 'float64'))

        result = clf.predict(hog_features)
    
    if int(result[0])<0:
        cv.putText(frame, "Predicted number: " , (5, 420),
                   cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)
    else:
        cv.putText(frame, "Predicted number: " + str(int(result)), (5, 420),
                   cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)
    

    cv.imshow('frame', frame)
    #cv.imshow('roi', roi)
    #cv.imshow('roi_image', roi_range)
    cv.imshow('board', board)
    cv.imshow('img', img)
    k = cv.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('c'):
        board.fill(0)
        centerpoints.clear()
        result = [-2, -1]
    # clearing the board
cap.release()
cv.destroyAllWindows()


