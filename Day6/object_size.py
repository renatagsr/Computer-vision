#To run the code
#python3 object_size.py --image /home/renata/Desktop/Cursos/projetos/Computer-vision/Images/example_03.png --width 0.955

#Import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

#Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the input image')
ap.add_argument('-w', '--width', type=float, required=True, help='Width of the left-most object in the image(in inches)')
args = vars(ap.parse_args())

#Load the image, convert it to grayscale and blur it slightly
image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

#Perfomr edge detection, then perform a dilation + erosin to close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

#Finde contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#Sort the contours from left-to-right and initialize the 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixels_per_metric = None

#Loop over the contours individually
for c in cnts:
    #If the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) < 100:
        continue

    #Compute the rotate bounding box of the contour
    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype='int')

    #Order the points in the contour such that they appear in top-left, top-right, bottom-right
    #and bottom-left order, then draw the outline of the rotated bounding box
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype('int')], -1, (0, 255, 0), 2)

    #Loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    #Unpack the ordered bounding box, then compute the midpoint between the top-left coordinates, followed by
    #the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    #Compute the midpoint between the top-left and top-right points, followed by the midpoint between
    #the top-right and bottom right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    #Draw the midpoints on the image
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    #Draw lines betweem the midpoints
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

    #Compute the Euclidean distace between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    #IF the pixes per metric has not been initialized, then compute it as the ratio of pixels 
    #to supplied metric (in this case, inches)
    if pixels_per_metric is None:
        pixels_per_metric = dB / args['width']

    #Compute the size of the object
    dimA = dA / pixels_per_metric
    dimB = dB / pixels_per_metric

    #Draw the object sizes on the image
    cv2.putText(orig, f'{dimA * 2.54:.1f} cm', (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 
                (255, 255, 255), 2)
    cv2.putText(orig, f'{dimB * 2.54:.1f} cm', (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 
                0.65, (255, 255, 255), 2)

    #Show the outpur image
    cv2.imshow('Image', orig)
    cv2.waitKey(0)