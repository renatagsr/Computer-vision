#Import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

#Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--prototxt', required=True, help='path to Caffe "deploy" prototxt file')
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#Load our serialized model from disk
print('[INFO] loading model...')
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

#Initialize the video stream and allow the camera sensor to warm up
print('[INFO] starting video stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0)

#Loop over the frames from the video stream
while True:
    #Grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=1200, height=12000)

    #Grab the fra dimension and convert it to a bloob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    #Pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    #Loop over the detections
    for i in range(0, detections.shape[2]):
        #Extract the confidence (i.e., probability) associated with the prdiciton 
        confidence = detections[0, 0, i, 2]

        #Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence < args['confidence']:
            continue
        
        #Compute the (x, y)-coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')

        #Draw the bounding box of the face along with the associated probability
        text = '{:.2f}%'.format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        #Show the output frame
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF

        #If the 'q' key was pressed, break from the loop
        if key == ord('q'):
            break

#Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()