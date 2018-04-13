import cv2
#import imutils
import numpy as np
#from hand import rangeHSV, joinMasks, findBiggestContour

image = cv2.imread('fachada.jpg')
height, width, channels = image.shape
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 40, 25, cv2.THRESH_BINARY)[1]
RANGE = 40
img_color = []

CENTRAL_AREA = 0.5
MIN_X = int(width * CENTRAL_AREA)
MAX_X = int(width * (1 - CENTRAL_AREA))
MIN_Y = int(height * CENTRAL_AREA)
MAX_Y = int(height * (1 - CENTRAL_AREA))

COORDENADAS = [(MAX_X, MIN_Y), (MAX_X, MAX_Y), (MIN_X, MIN_Y)]

for ponto in COORDENADAS:
    cv2.rectangle(image, ponto, (ponto[0]+4,ponto[1]+4), (0, 255, 0), 1)
    img_crop = image[ponto[1]:ponto[1] + 5, ponto[0]:ponto[0] + 5]
    img_color.append(cv2.cvtColor(np.uint8([[cv2.mean(img_crop)]]), cv2.COLOR_BGR2HSV))


imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = joinMasks(imghsv.copy(), img_color, RANGE)
W = findBiggestContour(mask.copy(), image)#busca o mair contorno da imagem

#find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cnts = cnts[0] if imutils.is_cv2() else cnts[1]



# # loop over the contours
for c in cnts:
    
     # compute the center of the contour
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])



    # # draw the contour and center of the shape on the image
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)


    # show imagem in a window
cv2.imshow('DETECCAO', image)

    # wait any key to close the window
cv2.waitKey(0)
