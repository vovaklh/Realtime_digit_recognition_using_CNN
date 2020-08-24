from imutils.perspective import four_point_transform
from imutils import contours, grab_contours
import cv2
import numpy as np


def four_point_transformation(image):
    # preprocess image
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    edged = cv2.Canny(image_blur, 70, 255)

    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)

    if cnts:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                return four_point_transform(image, approx.reshape(4, 2))
            else:
                return image
