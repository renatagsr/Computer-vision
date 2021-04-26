#What to put in the terminal to run the code
#python3 facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
#  --image /home/renata/Desktop/Cursos/projetos/Computer-vision/Images/Renata.jpg

#Import the necessary packages
from imutils import face_utils
import numpy as np
import argparse 
import imutils
import dlib
import cv2

#Construct the argumento parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-predictor', required=True, help='Path to facial landmark predictor')
ap.add_argument('-i', '--image', required=True, help='Path to input imae=ge')
args = vars(ap.parse_args())

#Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

#Load the input image, resize it, and convert it to grayscale
image = cv2.imread(args['image'])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Detect faces in the grayscale image
rects = detector(gray, 1)

#Loop over the face detections
for (i, rect) in enumerate(rects):
    #Determine the facial landmarks for the face region, then convert the facial
    #landmarka (x, y)-coordinates to a Numpy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    #Convert dlib's rectangle to a OpenCV-style bounding box 
    #[i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y+ h), (0, 255, 0), 2)

    #Show the face number 
    cv2.putText(image, 'Face#{}'.format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
    
    #Loop over the (x, y)-coordinates for the facial landmarks and 
    #Draw them on the image
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

#Show the output image with the face detections + facial landmarks
cv2.imshow('Output', image)
cv2.waitKey(0)

