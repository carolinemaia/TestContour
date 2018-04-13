import numpy as np
import cv2

def getMeans(img, w, h, points):
    # buscar uma forma de carregar os dados no momento certo
    median_points = []
    for dots in points:
        # Y after X
        temp_numpy = []
        crop_temp = img[dots[1]:(dots[1]+w), dots[0]:(dots[0]+h)]
        key_temp = 0
        for median in cv2.mean(crop_temp):
            if key_temp < 3:
                temp_numpy.append(int(median))
            key_temp += 1
        median_points.append(temp_numpy)
        #cv2.rectangle(img, dots, (dots[0]+10,dots[1]+10), (0, 255, 0), 1)
    return median_points

def convert2HSV(means):
    median_p = []
    for median in means:
        blbla = cv2.cvtColor(np.uint8([[median]]), cv2.COLOR_BGR2HSV)
        median_p.append(blbla)
    return median_p

def rangeHSV(hsv_median, sentido, rangecolor):
    color_temp = []
    key_temp = 0
    for hsvcolor in hsv_median[0][0]:
        if sentido == 'upper':
            hsvcolor += rangecolor
            if key_temp == 0 and hsvcolor > 180:
                hsvcolor = 180
            elif key_temp == 1 and hsvcolor > 255:
                hsvcolor = 255
            elif key_temp == 2 and hsvcolor > 255:
                hsvcolor = 255
            color_temp.append(hsvcolor)
            key_temp += 1
        if sentido == 'lower':
            hsvcolor -= rangecolor
            if hsvcolor < 0:
                hsvcolor = 0
            color_temp.append(hsvcolor)
    return color_temp

def joinMasks(imghsv, hsvmedians, ranges):
    kernel = np.ones((2, 2), np.uint8)
    temp_mask = np.zeros((imghsv.shape[0], imghsv.shape[1]), np.uint8)
    for hsvcolors in hsvmedians:
        hsvlower = rangeHSV(hsvcolors, 'lower', ranges)
        hsvupper = rangeHSV(hsvcolors, 'upper', ranges)
        mask = cv2.inRange(imghsv, np.uint8(hsvlower), np.uint8(hsvupper))
        #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        #mask = cv2.erode(mask, kernel, iterations=1)
        y = mask != 0
        temp_mask[y] = mask[y]
    return temp_mask

def findBiggestContour(mask, img):
    temp_bigger = []
    img1, cont, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in cont:
        temp_bigger.append(cv2.contourArea(cnt))
    greatest = max(temp_bigger)
    index_big = temp_bigger.index(greatest)
    key = 0
    for cnt in cont:
        if key == index_big:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x,y), (x+w,y+h), (0, 255, 0), 1)
            return w
            break
        key += 1
