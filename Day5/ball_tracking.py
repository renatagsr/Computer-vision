#Import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

#Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', 
                help='path to the (optional) video file')
ap.add_argument('-b', '--buffer', type=int, default=64,
                help='max buffer size')
args = vars(ap.parse_args())

#Define the lower and upper boundaries of the "green"
#ball in the HSV color space, then initialize the
#list of tracked points
grenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=args['buffer'])

#If a video path was not supplied, grab the reference
#to the webcam
if not args.get('video', False):
    vs = VideoStream(src=0).start()

#Otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args['video'])

#Allow the camera or video file to warm up 
time.sleep(2.0)

#Keep looping
while True:
    #Grab the current fram
    frame = vs.read()

    #Handle the fram from VideoCapture or VideoStream
    frame = frame[1] if args.get('video', False) else frame

    #If we are viewing a video and we did not grab a frame,
    #then we have reached the end of the video
    if frame is None:
        break
    
    #Resize the fram, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=1100, height=500)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    #Construct a mask for the color "green", then perform 
    #a series of dilations and erosions to remove any small
    #blobs left in the mask
    mask = cv2.inRange(hsv, grenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    #Find contours in the mask and initialize the current
    #(x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    #Only proceed if at least one contour was found
    if len(cnts) > 0:
        #Find the largest contour in the mask, then use
        #it to compute the minimum enclosing circle and
        #centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        #Only proceed if the radius meets a minimum size
        if radius >10:
            #Draw the circle and centroid on the frame,
            #then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    #Update the points queue
    pts.appendleft(center)

    #Loop over the set of tracked points
    for i in range(1, len(pts)):
        #If either of the tracked points are None,
        #ignore them
        if pts[i - 1] is None or pts[i] is None:
            continue
        
        #Otherwise, compute the thickness of the line and
        #draw the connecting lines
        thickness = int(np.sqrt(args['buffer'] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    #Show the fram to our screen
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    #IF the 'q' key is pressed, stop the loop
    if key == ord('q'):
        break

#If we are not using a video file, stop the camera video stream
if not args.get('video', False):
    vs.stop()

#Other wise, release the camera
else:
    vs.release()

#Close all windows
cv2.destroyAllWindows()

